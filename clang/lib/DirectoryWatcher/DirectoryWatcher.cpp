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

static timespec toTimeSpec(sys::TimePoint<> tp) {
  std::chrono::seconds sec = std::chrono::time_point_cast<std::chrono::seconds>(
                 tp).time_since_epoch();
  std::chrono::nanoseconds nsec =
    std::chrono::time_point_cast<std::chrono::nanoseconds>(tp - sec)
      .time_since_epoch();
  timespec ts;
  ts.tv_sec = sec.count();
  ts.tv_nsec = nsec.count();
  return ts;
}

static Optional<timespec> getModTime(StringRef path) {
  sys::fs::file_status Status;
  std::error_code EC = status(path, Status);
  if (EC)
    return None;
  return toTimeSpec(Status.getLastModificationTime());
}

struct DirectoryWatcher::Implementation {
#if HAVE_CORESERVICES
  FSEventStreamRef EventStream = nullptr;

  bool setupFSEventStream(StringRef path, EventReceiver receiver,
                          dispatch_queue_t queue);
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

  EventStreamContextData(std::string watchedPath, DirectoryWatcher::EventReceiver receiver)
  : WatchedPath(std::move(watchedPath)), Receiver(std::move(receiver)) {
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
        DirectoryWatcher::Event Evt{DirectoryWatcher::EventKind::DirectoryDeleted, path, timespec{}};
        Events.push_back(Evt);
        break;
      }
      continue;
    }
    DirectoryWatcher::EventKind K = DirectoryWatcher::EventKind::Modified;
    if ((flags & kFSEventStreamEventFlagItemCreated) ||
        (flags & kFSEventStreamEventFlagItemRenamed))
      K = DirectoryWatcher::EventKind::Added;
    if (flags & kFSEventStreamEventFlagItemRemoved)
      K = DirectoryWatcher::EventKind::Removed;
    timespec modTime{};
    if (K != DirectoryWatcher::EventKind::Removed) {
      auto modTimeOpt = getModTime(path);
      if (!modTimeOpt.hasValue())
        continue;
      modTime = modTimeOpt.getValue();
    }
    DirectoryWatcher::Event Evt{K, path, modTime};
    Events.push_back(Evt);
  }

  ctx->Receiver(Events, /*isInitial=*/false);
}

bool DirectoryWatcher::Implementation::setupFSEventStream(StringRef path,
                                                          EventReceiver receiver,
                                                          dispatch_queue_t queue) {
  if (path.empty())
    return true;

  CFMutableArrayRef pathsToWatch = CFArrayCreateMutable(nullptr, 0, &kCFTypeArrayCallBacks);
  CFStringRef cfPathStr = CFStringCreateWithBytes(nullptr, (const UInt8 *)path.data(), path.size(), kCFStringEncodingUTF8, false);
  CFArrayAppendValue(pathsToWatch, cfPathStr);
  CFRelease(cfPathStr);
  CFAbsoluteTime latency = 0.2; // Latency in seconds.

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

  EventStreamContextData *ctxData = new EventStreamContextData(std::move(realPath), std::move(receiver));
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

#if HAVE_CORESERVICES
static std::vector<DirectoryWatcher::Event> scanDirectory(StringRef Path) {
  using namespace llvm::sys;

  std::vector<DirectoryWatcher::Event> Events;
  std::error_code EC;
  for (auto It = fs::directory_iterator(Path, EC), End = fs::directory_iterator();
         !EC && It != End; It.increment(EC)) {
    auto modTime = getModTime(It->path());
    if (!modTime.hasValue())
      continue;
    DirectoryWatcher::Event Event{DirectoryWatcher::EventKind::Added, It->path(), modTime.getValue()};
    Events.push_back(std::move(Event));
  }
  return Events;
}
#endif

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
    auto events = scanDirectory(copiedPath);
    Receiver(events, /*isInitial=*/true);
    dispatch_semaphore_signal(initScanSema);
    dispatch_release(setupFSEventsSema);
    dispatch_release(initScanSema);
  });
  bool fsErr = Impl.setupFSEventStream(Path, Receiver, queue);
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
