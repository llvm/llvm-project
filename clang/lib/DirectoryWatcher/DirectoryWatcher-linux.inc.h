//===- DirectoryWatcher-linux.inc.h - Linux-platform directory listening --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Errno.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/Path.h"
#include <sys/inotify.h>
#include <thread>
#include <unistd.h>

namespace {

struct INotifyEvent {
  DirectoryWatcher::EventKind K;
  std::string Filename;
  Optional<sys::fs::file_status> Status;
};

class EventQueue {
  DirectoryWatcher::EventReceiver Receiver;
  sys::Mutex Mtx;
  bool gotInitialScan = false;
  std::vector<INotifyEvent> PendingEvents;

  DirectoryWatcher::Event toDirEvent(const INotifyEvent &evt) {
    llvm::sys::TimePoint<> modTime{};
    if (evt.Status.hasValue())
      modTime = evt.Status->getLastModificationTime();
    return DirectoryWatcher::Event{evt.K, evt.Filename, modTime};
  }

public:
  explicit EventQueue(DirectoryWatcher::EventReceiver receiver)
      : Receiver(receiver) {}

  void onDirectoryEvents(ArrayRef<INotifyEvent> evts) {
    sys::ScopedLock L(Mtx);

    if (!gotInitialScan) {
      PendingEvents.insert(PendingEvents.end(), evts.begin(), evts.end());
      return;
    }

    SmallVector<DirectoryWatcher::Event, 8> dirEvents;
    for (const auto &evt : evts) {
      dirEvents.push_back(toDirEvent(evt));
    }
    Receiver(dirEvents, /*isInitial=*/false);
  }

  void onInitialScan(std::shared_ptr<DirectoryScan> dirScan) {
    sys::ScopedLock L(Mtx);

    std::vector<DirectoryWatcher::Event> events = dirScan->getAsFileEvents();
    Receiver(events, /*isInitial=*/true);

    events.clear();
    for (const auto &evt : PendingEvents) {
      if (evt.K == DirectoryWatcher::EventKind::Added &&
          dirScan->FileIDSet.count(evt.Status->getUniqueID())) {
        // Already reported this event at the initial directory scan.
        continue;
      }
      events.push_back(toDirEvent(evt));
    }
    if (!events.empty()) {
      Receiver(events, /*isInitial=*/false);
    }

    gotInitialScan = true;
    PendingEvents.clear();
  }
};
} // namespace

struct DirectoryWatcher::Implementation {
  bool initialize(StringRef Path, EventReceiver Receiver, bool waitInitialSync,
                  std::string &Error);
  ~Implementation() { stopListening(); };

private:
  int inotifyFD = -1;

  void stopListening();
};

static void runWatcher(std::string pathToWatch, int inotifyFD,
                       std::shared_ptr<EventQueue> evtQueue) {
#define EVT_BUF_LEN (30 * (sizeof(struct inotify_event) + NAME_MAX + 1))
  char buf[EVT_BUF_LEN] __attribute__((aligned(8)));

  while (1) {
    ssize_t numRead = read(inotifyFD, buf, EVT_BUF_LEN);
    if (numRead == -1) {
      return; // watcher is stopped.
    }

    SmallVector<INotifyEvent, 8> iEvents;
    for (char *p = buf; p < buf + numRead;) {
      struct inotify_event *ievt = (struct inotify_event *)p;
      p += sizeof(struct inotify_event) + ievt->len;

      if (ievt->mask & IN_DELETE_SELF) {
        INotifyEvent iEvt{DirectoryWatcher::EventKind::DirectoryDeleted,
                          pathToWatch, None};
        iEvents.push_back(iEvt);
        break;
      }

      DirectoryWatcher::EventKind K = DirectoryWatcher::EventKind::Added;
      if (ievt->mask & IN_MODIFY) {
        K = DirectoryWatcher::EventKind::Modified;
      }
      if (ievt->mask & IN_MOVED_TO) {
        K = DirectoryWatcher::EventKind::Added;
      }
      if (ievt->mask & IN_DELETE) {
        K = DirectoryWatcher::EventKind::Removed;
      }

      assert(ievt->len > 0 && "expected a filename from inotify");
      SmallString<256> fullPath{pathToWatch};
      sys::path::append(fullPath, ievt->name);

      Optional<sys::fs::file_status> statusOpt;
      if (K != DirectoryWatcher::EventKind::Removed) {
        statusOpt = getFileStatus(fullPath);
        if (!statusOpt.hasValue())
          K = DirectoryWatcher::EventKind::Removed;
      }
      INotifyEvent iEvt{K, fullPath.str(), statusOpt};
      iEvents.push_back(iEvt);
    }

    if (!iEvents.empty())
      evtQueue->onDirectoryEvents(iEvents);
  }
}

bool DirectoryWatcher::Implementation::initialize(StringRef Path,
                                                  EventReceiver Receiver,
                                                  bool waitInitialSync,
                                                  std::string &errorMsg) {
  auto error = [&](StringRef msg) -> bool {
    errorMsg = msg;
    errorMsg += ": ";
    errorMsg += llvm::sys::StrError();
    return true;
  };

  auto evtQueue = std::make_shared<EventQueue>(std::move(Receiver));

  inotifyFD = inotify_init();
  if (inotifyFD == -1)
    return error("inotify_init failed");

  std::string pathToWatch = Path;
  int wd = inotify_add_watch(inotifyFD, pathToWatch.c_str(),
                             IN_MOVED_TO | IN_DELETE | IN_MODIFY |
                                 IN_DELETE_SELF | IN_ONLYDIR);
  if (wd == -1)
    return error("inotify_add_watch failed");

  std::thread watchThread(
      std::bind(runWatcher, pathToWatch, inotifyFD, evtQueue));
  watchThread.detach();

  auto initialScan = std::make_shared<DirectoryScan>();
  auto runScan = [pathToWatch, initialScan, evtQueue]() {
    initialScan->scanDirectory(pathToWatch);
    evtQueue->onInitialScan(std::move(initialScan));
  };

  if (waitInitialSync) {
    runScan();
  } else {
    std::thread scanThread(runScan);
    scanThread.detach();
  }

  return false;
}

void DirectoryWatcher::Implementation::stopListening() {
  if (inotifyFD == -1)
    return;
  close(inotifyFD);
  inotifyFD = -1;
}
