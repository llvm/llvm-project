//===- DirectoryWatcher-mac.inc.h - Mac-platform directory listening ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <CoreServices/CoreServices.h>

struct DirectoryWatcher::Implementation {
  bool initialize(StringRef Path, EventReceiver Receiver, bool waitInitialSync,
                  std::string &Error);
  ~Implementation() { stopFSEventStream(); };

private:
  FSEventStreamRef EventStream = nullptr;

  bool setupFSEventStream(StringRef path, EventReceiver receiver,
                          dispatch_queue_t queue,
                          std::shared_ptr<DirectoryScan> initialScanPtr);
  void stopFSEventStream();
};

namespace {
struct EventStreamContextData {
  std::string WatchedPath;
  DirectoryWatcher::EventReceiver Receiver;
  std::shared_ptr<DirectoryScan> InitialScan;

  EventStreamContextData(std::string watchedPath,
                         DirectoryWatcher::EventReceiver receiver,
                         std::shared_ptr<DirectoryScan> initialScanPtr)
      : WatchedPath(std::move(watchedPath)), Receiver(std::move(receiver)),
        InitialScan(std::move(initialScanPtr)) {}

  static void dispose(const void *ctx) {
    delete static_cast<const EventStreamContextData *>(ctx);
  }
};
} // namespace

static void eventStreamCallback(ConstFSEventStreamRef stream,
                                void *clientCallBackInfo, size_t numEvents,
                                void *eventPaths,
                                const FSEventStreamEventFlags eventFlags[],
                                const FSEventStreamEventId eventIds[]) {
  auto *ctx = static_cast<EventStreamContextData *>(clientCallBackInfo);

  std::vector<DirectoryWatcher::Event> Events;
  for (size_t i = 0; i < numEvents; ++i) {
    StringRef path = ((const char **)eventPaths)[i];
    const FSEventStreamEventFlags flags = eventFlags[i];
    if (!(flags & kFSEventStreamEventFlagItemIsFile)) {
      if ((flags & kFSEventStreamEventFlagItemRemoved) &&
          path == ctx->WatchedPath) {
        DirectoryWatcher::Event Evt{
            DirectoryWatcher::EventKind::DirectoryDeleted, path,
            llvm::sys::TimePoint<>{}};
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

bool DirectoryWatcher::Implementation::setupFSEventStream(
    StringRef path, EventReceiver receiver, dispatch_queue_t queue,
    std::shared_ptr<DirectoryScan> initialScanPtr) {
  if (path.empty())
    return true;

  CFMutableArrayRef pathsToWatch =
      CFArrayCreateMutable(nullptr, 0, &kCFTypeArrayCallBacks);
  CFStringRef cfPathStr =
      CFStringCreateWithBytes(nullptr, (const UInt8 *)path.data(), path.size(),
                              kCFStringEncodingUTF8, false);
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

  EventStreamContextData *ctxData = new EventStreamContextData(
      std::move(realPath), std::move(receiver), std::move(initialScanPtr));
  FSEventStreamContext context;
  context.version = 0;
  context.info = ctxData;
  context.retain = nullptr;
  context.release = EventStreamContextData::dispose;
  context.copyDescription = nullptr;

  EventStream = FSEventStreamCreate(
      nullptr, eventStreamCallback, &context, pathsToWatch,
      kFSEventStreamEventIdSinceNow, latency,
      kFSEventStreamCreateFlagFileEvents | kFSEventStreamCreateFlagNoDefer);
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

bool DirectoryWatcher::Implementation::initialize(StringRef Path,
                                                  EventReceiver Receiver,
                                                  bool waitInitialSync,
                                                  std::string &Error) {
  auto initialScan = std::make_shared<DirectoryScan>();

  dispatch_queue_t queue =
      dispatch_queue_create("DirectoryWatcher", DISPATCH_QUEUE_SERIAL);
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
  bool fsErr = setupFSEventStream(Path, Receiver, queue, initialScan);
  dispatch_semaphore_signal(setupFSEventsSema);

  if (waitInitialSync) {
    dispatch_semaphore_wait(initScanSema, DISPATCH_TIME_FOREVER);
  }
  dispatch_release(setupFSEventsSema);
  dispatch_release(initScanSema);
  dispatch_release(queue);

  if (fsErr) {
    raw_string_ostream(Error)
        << "failed to setup FSEvents stream for path: " << Path;
    return true;
  }

  return false;
}
