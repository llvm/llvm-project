//===- DirectoryWatcher.cpp - Listens for directory file changes ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// \brief Utility class for listening for file system changes in a directory.
//===----------------------------------------------------------------------===//

#include "clang/DirectoryWatcher/DirectoryWatcher.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#define HAVE_CORESERVICES 0

#if defined(__has_include)
#if __has_include(<CoreServices/CoreServices.h>)

#include <CoreServices/CoreServices.h>
#undef HAVE_CORESERVICES
#define HAVE_CORESERVICES 1

#endif
#endif

using namespace clang;
using namespace llvm;

static Optional<sys::fs::file_status> getFileStatus(StringRef path) {
  sys::fs::file_status Status;
  std::error_code EC = status(path, Status);
  if (EC)
    return None;
  return Status;
}

namespace llvm {
// Specialize DenseMapInfo for sys::fs::UniqueID.
template <> struct DenseMapInfo<sys::fs::UniqueID> {
  static sys::fs::UniqueID getEmptyKey() {
    return sys::fs::UniqueID{DenseMapInfo<uint64_t>::getEmptyKey(),
                             DenseMapInfo<uint64_t>::getEmptyKey()};
  }

  static sys::fs::UniqueID getTombstoneKey() {
    return sys::fs::UniqueID{DenseMapInfo<uint64_t>::getTombstoneKey(),
                             DenseMapInfo<uint64_t>::getEmptyKey()};
  }

  static unsigned getHashValue(const sys::fs::UniqueID &val) {
    return DenseMapInfo<std::pair<uint64_t, uint64_t>>::getHashValue(
        std::make_pair(val.getDevice(), val.getFile()));
  }

  static bool isEqual(const sys::fs::UniqueID &LHS, const sys::fs::UniqueID &RHS) {
    return LHS == RHS;
  }
};
}

namespace {
/// Used for initial directory scan.
///
/// Note that it is only accessed while inside the serial queue so it is thread
/// safe to access it without additional protection.
struct DirectoryScan {
  DenseSet<sys::fs::UniqueID> FileIDSet;
  std::vector<std::tuple<std::string, sys::TimePoint<>>> Files;

  void scanDirectory(StringRef Path) {
    using namespace llvm::sys;

    std::error_code EC;
    for (auto It = fs::directory_iterator(Path, EC), End = fs::directory_iterator();
           !EC && It != End; It.increment(EC)) {
      auto status = getFileStatus(It->path());
      if (!status.hasValue())
        continue;
      Files.push_back(std::make_tuple(It->path(), status->getLastModificationTime()));
      FileIDSet.insert(status->getUniqueID());
    }
  }

  std::vector<DirectoryWatcher::Event> getAsFileEvents() const {
    std::vector<DirectoryWatcher::Event> Events;
    for (const auto &info : Files) {
      DirectoryWatcher::Event Event{DirectoryWatcher::EventKind::Added, std::get<0>(info), std::get<1>(info)};
      Events.push_back(std::move(Event));
    }
    return Events;
  }
};
}

struct DirectoryWatcher::Implementation {
#if HAVE_CORESERVICES
  FSEventStreamRef EventStream = nullptr;

  bool setupFSEventStream(StringRef path, EventReceiver receiver,
                          dispatch_queue_t queue,
                          std::shared_ptr<DirectoryScan> initialScanPtr);
  void stopFSEventStream();

  ~Implementation() {
    stopFSEventStream();
  };
#endif
};

#if HAVE_CORESERVICES
namespace {
struct EventStreamContextData {
  std::string WatchedPath;
  DirectoryWatcher::EventReceiver Receiver;
  std::shared_ptr<DirectoryScan> InitialScan;

  EventStreamContextData(std::string watchedPath, DirectoryWatcher::EventReceiver receiver,
                         std::shared_ptr<DirectoryScan> initialScanPtr)
  : WatchedPath(std::move(watchedPath)),
    Receiver(std::move(receiver)),
    InitialScan(std::move(initialScanPtr)) {
  }

  static void dispose(const void *ctx) {
    delete static_cast<const EventStreamContextData*>(ctx);
  }
};
}

static void eventStreamCallback(
                       ConstFSEventStreamRef stream,
                       void *clientCallBackInfo,
                       size_t numEvents,
                       void *eventPaths,
                       const FSEventStreamEventFlags eventFlags[],
                       const FSEventStreamEventId eventIds[]) {
  auto *ctx = static_cast<EventStreamContextData*>(clientCallBackInfo);

  std::vector<DirectoryWatcher::Event> Events;
  for (size_t i = 0; i < numEvents; ++i) {
    StringRef path = ((const char **)eventPaths)[i];
    const FSEventStreamEventFlags flags = eventFlags[i];
    if (!(flags & kFSEventStreamEventFlagItemIsFile)) {
      if ((flags & kFSEventStreamEventFlagItemRemoved) && path == ctx->WatchedPath) {
        DirectoryWatcher::Event Evt{DirectoryWatcher::EventKind::DirectoryDeleted, path, llvm::sys::TimePoint<>{} };
        Events.push_back(Evt);
        break;
      }
      continue;
    }
    DirectoryWatcher::EventKind K = DirectoryWatcher::EventKind::Modified;
    bool hasAddedFlag = flags & (kFSEventStreamEventFlagItemCreated |
                                 kFSEventStreamEventFlagItemRenamed);
    bool hasRemovedFlag = flags & kFSEventStreamEventFlagItemRemoved;
    Optional<sys::fs::file_status> statusOpt;
    // NOTE: With low latency sometimes for a file that is moved inside the
    // directory, or for a file that is removed from the directory, the flags
    // have both 'renamed' and 'removed'. We use getting the file status as a
    // way to distinguish between the two.
    if (hasAddedFlag) {
      statusOpt = getFileStatus(path);
      if (statusOpt.hasValue()) {
        K = DirectoryWatcher::EventKind::Added;
      } else {
        K = DirectoryWatcher::EventKind::Removed;
      }
    } else if (hasRemovedFlag) {
      K = DirectoryWatcher::EventKind::Removed;
    } else {
      statusOpt = getFileStatus(path);
      if (!statusOpt.hasValue()) {
        K = DirectoryWatcher::EventKind::Removed;
      }
    }

    if (ctx->InitialScan && K == DirectoryWatcher::EventKind::Added) {
      // For the first time we get the events, check that we haven't already
      // sent the 'added' event at the initial scan.
      if (ctx->InitialScan->FileIDSet.count(statusOpt->getUniqueID())) {
        // Already reported this event at the initial directory scan.
        continue;
      }
    }

    llvm::sys::TimePoint<> modTime{};
    if (statusOpt.hasValue())
      modTime = statusOpt->getLastModificationTime();
    DirectoryWatcher::Event Evt{K, path, modTime};
    Events.push_back(Evt);
  }

  // We won't need to check again later on.
  ctx->InitialScan.reset();

  if (!Events.empty()) {
    ctx->Receiver(Events, /*isInitial=*/false);
  }
}

bool DirectoryWatcher::Implementation::setupFSEventStream(StringRef path,
                                                          EventReceiver receiver,
                                                          dispatch_queue_t queue,
                                                          std::shared_ptr<DirectoryScan> initialScanPtr) {
  if (path.empty())
    return true;

  CFMutableArrayRef pathsToWatch = CFArrayCreateMutable(nullptr, 0, &kCFTypeArrayCallBacks);
  CFStringRef cfPathStr = CFStringCreateWithBytes(nullptr, (const UInt8 *)path.data(), path.size(), kCFStringEncodingUTF8, false);
  CFArrayAppendValue(pathsToWatch, cfPathStr);
  CFRelease(cfPathStr);
  CFAbsoluteTime latency = 0.0; // Latency in seconds.

  std::string realPath;
  {
    SmallString<128> Storage;
    StringRef P = llvm::Twine(path).toNullTerminatedStringRef(Storage);
    char Buffer[PATH_MAX];
    // Use ::realpath to get the real path name
    if (::realpath(P.begin(), Buffer) != nullptr)
      realPath = Buffer;
    else
      realPath = path;
  }

  EventStreamContextData *ctxData =
    new EventStreamContextData(std::move(realPath), std::move(receiver),
                               std::move(initialScanPtr));
  FSEventStreamContext context;
  context.version = 0;
  context.info = ctxData;
  context.retain = nullptr;
  context.release = EventStreamContextData::dispose;
  context.copyDescription = nullptr;

  EventStream = FSEventStreamCreate(nullptr,
                                    eventStreamCallback,
                                    &context,
                                    pathsToWatch,
                                    kFSEventStreamEventIdSinceNow,
                                    latency,
                                    kFSEventStreamCreateFlagFileEvents |
                                    kFSEventStreamCreateFlagNoDefer);
  CFRelease(pathsToWatch);
  if (!EventStream) {
    return true;
  }
  FSEventStreamSetDispatchQueue(EventStream, queue);
  FSEventStreamStart(EventStream);
  return false;
}

void DirectoryWatcher::Implementation::stopFSEventStream() {
  if (!EventStream)
    return;
  FSEventStreamStop(EventStream);
  FSEventStreamInvalidate(EventStream);
  FSEventStreamRelease(EventStream);
  EventStream = nullptr;
}
#endif

DirectoryWatcher::DirectoryWatcher()
  : Impl(*new Implementation()) {}

DirectoryWatcher::~DirectoryWatcher() {
  delete &Impl;
}

std::unique_ptr<DirectoryWatcher> DirectoryWatcher::create(StringRef Path,
        EventReceiver Receiver, bool waitInitialSync, std::string &Error) {
#if HAVE_CORESERVICES

  using namespace llvm::sys;

  if (!fs::exists(Path)) {
    std::error_code EC = fs::create_directories(Path);
    if (EC) {
      Error = EC.message();
      return nullptr;
    }
  }

  bool IsDir;
  std::error_code EC = fs::is_directory(Path, IsDir);
  if (EC) {
    Error = EC.message();
    return nullptr;
  }
  if (!IsDir) {
    Error = "path is not a directory: ";
    Error += Path;
    return nullptr;
  }

  std::unique_ptr<DirectoryWatcher> DirWatch;
  DirWatch.reset(new DirectoryWatcher());
  auto &Impl = DirWatch->Impl;

  auto initialScan = std::make_shared<DirectoryScan>();

  dispatch_queue_t queue = dispatch_queue_create("DirectoryWatcher", DISPATCH_QUEUE_SERIAL);
  dispatch_semaphore_t initScanSema = dispatch_semaphore_create(0);
  dispatch_semaphore_t setupFSEventsSema = dispatch_semaphore_create(0);

  std::string copiedPath = Path;
  dispatch_retain(initScanSema);
  dispatch_retain(setupFSEventsSema);
  dispatch_async(queue, ^{
    // Wait for the event stream to be setup before doing the initial scan,
    // to make sure we won't miss any events.
    dispatch_semaphore_wait(setupFSEventsSema, DISPATCH_TIME_FOREVER);
    initialScan->scanDirectory(copiedPath);
    Receiver(initialScan->getAsFileEvents(), /*isInitial=*/true);
    dispatch_semaphore_signal(initScanSema);
    dispatch_release(setupFSEventsSema);
    dispatch_release(initScanSema);
  });
  bool fsErr = Impl.setupFSEventStream(Path, Receiver, queue, initialScan);
  dispatch_semaphore_signal(setupFSEventsSema);

  if (waitInitialSync) {
    dispatch_semaphore_wait(initScanSema, DISPATCH_TIME_FOREVER);
  }
  dispatch_release(setupFSEventsSema);
  dispatch_release(initScanSema);
  dispatch_release(queue);

  if (fsErr) {
    raw_string_ostream(Error) << "failed to setup FSEvents stream for path: " << Path;
    return nullptr;
  }

  return DirWatch;
#else
  return nullptr;
#endif
}
