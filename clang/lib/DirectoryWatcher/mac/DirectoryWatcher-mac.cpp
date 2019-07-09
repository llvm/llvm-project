//===- DirectoryWatcher-mac.cpp - Mac-platform directory watching ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DirectoryScanner.h"
#include "clang/DirectoryWatcher/DirectoryWatcher.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"
#include <CoreServices/CoreServices.h>

using namespace llvm;
using namespace clang;

static FSEventStreamRef createFSEventStream(
    StringRef Path,
    std::function<void(llvm::ArrayRef<DirectoryWatcher::Event>, bool)>,
    dispatch_queue_t);
static void stopFSEventStream(FSEventStreamRef);

namespace {

class DirectoryWatcherMac : public clang::DirectoryWatcher {
public:
  DirectoryWatcherMac(
      FSEventStreamRef EventStream,
      std::function<void(llvm::ArrayRef<DirectoryWatcher::Event>, bool)>
          Receiver,
      llvm::StringRef WatchedDirPath)
      : EventStream(EventStream), Receiver(Receiver),
        WatchedDirPath(WatchedDirPath) {}

  ~DirectoryWatcherMac() override {
    stopFSEventStream(EventStream);
    EventStream = nullptr;
    // Now it's safe to use Receiver as the only other concurrent use would have
    // been in EventStream processing.
    Receiver(DirectoryWatcher::Event(
                 DirectoryWatcher::Event::EventKind::WatcherGotInvalidated, ""),
             false);
  }

private:
  FSEventStreamRef EventStream;
  std::function<void(llvm::ArrayRef<Event>, bool)> Receiver;
  const std::string WatchedDirPath;
};

struct EventStreamContextData {
  std::string WatchedPath;
  std::function<void(llvm::ArrayRef<DirectoryWatcher::Event>, bool)> Receiver;

  EventStreamContextData(
      std::string &&WatchedPath,
      std::function<void(llvm::ArrayRef<DirectoryWatcher::Event>, bool)>
          Receiver)
      : WatchedPath(std::move(WatchedPath)), Receiver(Receiver) {}

  // Needed for FSEvents
  static void dispose(const void *ctx) {
    delete static_cast<const EventStreamContextData *>(ctx);
  }
};
} // namespace

constexpr const FSEventStreamEventFlags StreamInvalidatingFlags =
    kFSEventStreamEventFlagUserDropped | kFSEventStreamEventFlagKernelDropped |
    kFSEventStreamEventFlagMustScanSubDirs;

constexpr const FSEventStreamEventFlags ModifyingFileEvents =
    kFSEventStreamEventFlagItemCreated | kFSEventStreamEventFlagItemRenamed |
    kFSEventStreamEventFlagItemModified;

static void eventStreamCallback(ConstFSEventStreamRef Stream,
                                void *ClientCallBackInfo, size_t NumEvents,
                                void *EventPaths,
                                const FSEventStreamEventFlags EventFlags[],
                                const FSEventStreamEventId EventIds[]) {
  auto *ctx = static_cast<EventStreamContextData *>(ClientCallBackInfo);

  std::vector<DirectoryWatcher::Event> Events;
  for (size_t i = 0; i < NumEvents; ++i) {
    StringRef Path = ((const char **)EventPaths)[i];
    const FSEventStreamEventFlags Flags = EventFlags[i];

    if (Flags & StreamInvalidatingFlags) {
      Events.emplace_back(DirectoryWatcher::Event{
          DirectoryWatcher::Event::EventKind::WatcherGotInvalidated, ""});
      break;
    } else if (!(Flags & kFSEventStreamEventFlagItemIsFile)) {
      // Subdirectories aren't supported - if some directory got removed it
      // must've been the watched directory itself.
      if ((Flags & kFSEventStreamEventFlagItemRemoved) &&
          Path == ctx->WatchedPath) {
        Events.emplace_back(DirectoryWatcher::Event{
            DirectoryWatcher::Event::EventKind::WatchedDirRemoved, ""});
        Events.emplace_back(DirectoryWatcher::Event{
            DirectoryWatcher::Event::EventKind::WatcherGotInvalidated, ""});
        break;
      }
      // No support for subdirectories - just ignore everything.
      continue;
    } else if (Flags & kFSEventStreamEventFlagItemRemoved) {
      Events.emplace_back(DirectoryWatcher::Event::EventKind::Removed,
                          llvm::sys::path::filename(Path));
      continue;
    } else if (Flags & ModifyingFileEvents) {
      if (!getFileStatus(Path).hasValue()) {
        Events.emplace_back(DirectoryWatcher::Event::EventKind::Removed,
                            llvm::sys::path::filename(Path));
      } else {
        Events.emplace_back(DirectoryWatcher::Event::EventKind::Modified,
                            llvm::sys::path::filename(Path));
      }
      continue;
    }

    // default
    Events.emplace_back(DirectoryWatcher::Event{
        DirectoryWatcher::Event::EventKind::WatcherGotInvalidated, ""});
    llvm_unreachable("Unknown FSEvent type.");
  }

  if (!Events.empty()) {
    ctx->Receiver(Events, /*IsInitial=*/false);
  }
}

FSEventStreamRef createFSEventStream(
    StringRef Path,
    std::function<void(llvm::ArrayRef<DirectoryWatcher::Event>, bool)> Receiver,
    dispatch_queue_t Queue) {
  if (Path.empty())
    return nullptr;

  CFMutableArrayRef PathsToWatch = [&]() {
    CFMutableArrayRef PathsToWatch =
        CFArrayCreateMutable(nullptr, 0, &kCFTypeArrayCallBacks);
    CFStringRef CfPathStr =
        CFStringCreateWithBytes(nullptr, (const UInt8 *)Path.data(),
                                Path.size(), kCFStringEncodingUTF8, false);
    CFArrayAppendValue(PathsToWatch, CfPathStr);
    CFRelease(CfPathStr);
    return PathsToWatch;
  }();

  FSEventStreamContext Context = [&]() {
    std::string RealPath;
    {
      SmallString<128> Storage;
      StringRef P = llvm::Twine(Path).toNullTerminatedStringRef(Storage);
      char Buffer[PATH_MAX];
      if (::realpath(P.begin(), Buffer) != nullptr)
        RealPath = Buffer;
      else
        RealPath = Path;
    }

    FSEventStreamContext Context;
    Context.version = 0;
    Context.info = new EventStreamContextData(std::move(RealPath), Receiver);
    Context.retain = nullptr;
    Context.release = EventStreamContextData::dispose;
    Context.copyDescription = nullptr;
    return Context;
  }();

  FSEventStreamRef Result = FSEventStreamCreate(
      nullptr, eventStreamCallback, &Context, PathsToWatch,
      kFSEventStreamEventIdSinceNow, /* latency in seconds */ 0.0,
      kFSEventStreamCreateFlagFileEvents | kFSEventStreamCreateFlagNoDefer);
  CFRelease(PathsToWatch);

  return Result;
}

void stopFSEventStream(FSEventStreamRef EventStream) {
  if (!EventStream)
    return;
  FSEventStreamStop(EventStream);
  FSEventStreamInvalidate(EventStream);
  FSEventStreamRelease(EventStream);
}

std::unique_ptr<DirectoryWatcher> clang::DirectoryWatcher::create(
    StringRef Path,
    std::function<void(llvm::ArrayRef<DirectoryWatcher::Event>, bool)> Receiver,
    bool WaitForInitialSync) {
  dispatch_queue_t Queue =
      dispatch_queue_create("DirectoryWatcher", DISPATCH_QUEUE_SERIAL);

  if (Path.empty())
    return nullptr;

  auto EventStream = createFSEventStream(Path, Receiver, Queue);
  if (!EventStream) {
    return nullptr;
  }

  std::unique_ptr<DirectoryWatcher> Result =
      llvm::make_unique<DirectoryWatcherMac>(EventStream, Receiver, Path);

  // We need to copy the data so the lifetime is ok after a const copy is made
  // for the block.
  const std::string CopiedPath = Path;

  auto InitWork = ^{
    // We need to start watching the directory before we start scanning in order
    // to not miss any event. By dispatching this on the same serial Queue as
    // the FSEvents will be handled we manage to start watching BEFORE the
    // inital scan and handling events ONLY AFTER the scan finishes.
    FSEventStreamSetDispatchQueue(EventStream, Queue);
    FSEventStreamStart(EventStream);
    // We need to decrement the ref count for Queue as initialize() will return
    // and FSEvents has incremented it. Since we have to wait for FSEvents to
    // take ownership it's the easiest to do it here rather than main thread.
    dispatch_release(Queue);
    Receiver(getAsFileEvents(scanDirectory(CopiedPath)), /*IsInitial=*/true);
  };

  if (WaitForInitialSync) {
    dispatch_sync(Queue, InitWork);
  } else {
    dispatch_async(Queue, InitWork);
  }

  return Result;
}
