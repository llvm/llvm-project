//===- unittests/DirectoryWatcher/DirectoryWatcherTest.cpp ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/DirectoryWatcher/DirectoryWatcher.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "gtest/gtest.h"
#include <condition_variable>
#include <mutex>
#include <thread>

using namespace llvm;
using namespace llvm::sys;
using namespace llvm::sys::fs;
using namespace clang;

namespace {

class EventCollection {
  SmallVector<DirectoryWatcher::Event, 6> Events;

public:
  EventCollection() = default;
  explicit EventCollection(ArrayRef<DirectoryWatcher::Event> events) {
    append(events);
  }

  void append(ArrayRef<DirectoryWatcher::Event> events) {
    Events.append(events.begin(), events.end());
  }

  bool empty() const { return Events.empty(); }
  size_t size() const { return Events.size(); }
  void clear() { Events.clear(); }

  bool hasEvents(ArrayRef<StringRef> filenames,
                 ArrayRef<DirectoryWatcher::EventKind> kinds,
                 ArrayRef<file_status> stats) const {
    assert(filenames.size() == kinds.size());
    assert(filenames.size() == stats.size());
    SmallVector<DirectoryWatcher::Event, 6> evts = Events;
    bool hadError = false;
    for (unsigned i = 0, e = filenames.size(); i < e; ++i) {
      StringRef fname = filenames[i];
      DirectoryWatcher::EventKind kind = kinds[i];
      file_status stat = stats[i];
      auto it = std::find_if(evts.begin(), evts.end(),
                             [&](const DirectoryWatcher::Event &evt) -> bool {
                               return path::filename(evt.Filename) == fname;
                             });
      if (it == evts.end()) {
        hadError = err(Twine("expected filename '" + fname + "' not found"));
        continue;
      }
      if (it->Kind != kind) {
        hadError = err(Twine("filename '" + fname + "' has event kind " +
                             std::to_string((int)it->Kind) + ", expected ") +
                       std::to_string((int)kind));
      }
      if (it->Kind != DirectoryWatcher::EventKind::Removed &&
          it->ModTime != stat.getLastModificationTime())
        hadError =
            err(Twine("filename '" + fname + "' has different mod time"));
      evts.erase(it);
    }
    for (const auto &evt : evts) {
      hadError = err(Twine("unexpected filename '" +
                           path::filename(evt.Filename) + "' found"));
    }
    return !hadError;
  }

  bool hasAdded(ArrayRef<StringRef> filenames,
                ArrayRef<file_status> stats) const {
    std::vector<DirectoryWatcher::EventKind> kinds{
        filenames.size(), DirectoryWatcher::EventKind::Added};
    return hasEvents(filenames, kinds, stats);
  }

  bool hasRemoved(ArrayRef<StringRef> filenames) const {
    std::vector<DirectoryWatcher::EventKind> kinds{
        filenames.size(), DirectoryWatcher::EventKind::Removed};
    std::vector<file_status> stats{filenames.size(), file_status{}};
    return hasEvents(filenames, kinds, stats);
  }

private:
  bool err(Twine msg) const {
    SmallString<128> buf;
    llvm::errs() << msg.toStringRef(buf) << '\n';
    return true;
  }
};

struct EventOccurrence {
  std::vector<DirectoryWatcher::Event> Events;
  bool IsInitial;
};

class DirectoryWatcherTest
    : public std::enable_shared_from_this<DirectoryWatcherTest> {
  std::string WatchedDir;
  std::string TempDir;
  std::unique_ptr<DirectoryWatcher> DirWatcher;

  std::condition_variable Condition;
  std::mutex Mutex;
  std::deque<EventOccurrence> EvtOccurs;

public:
  void init() {
    SmallString<128> pathBuf;
    std::error_code EC = createUniqueDirectory("dirwatcher", pathBuf);
    ASSERT_FALSE(EC);
    TempDir = pathBuf.str();
    path::append(pathBuf, "watch");
    WatchedDir = pathBuf.str();
    EC = create_directory(WatchedDir);
    ASSERT_FALSE(EC);
  }

  ~DirectoryWatcherTest() {
    stopWatching();
    remove_directories(TempDir);
  }

public:
  StringRef getWatchedDir() const { return WatchedDir; }

  void addFile(StringRef filename, file_status &stat) {
    SmallString<128> pathBuf;
    pathBuf = TempDir;
    path::append(pathBuf, filename);
    Expected<file_t> ft =
        openNativeFileForWrite(pathBuf, CD_CreateNew, OF_None);
    ASSERT_TRUE((bool)ft);
    closeFile(*ft);

    SmallString<128> newPath;
    newPath = WatchedDir;
    path::append(newPath, filename);
    std::error_code EC = rename(pathBuf, newPath);
    ASSERT_FALSE(EC);

    EC = status(newPath, stat);
    ASSERT_FALSE(EC);
  }

  void addFiles(ArrayRef<StringRef> filenames,
                std::vector<file_status> &stats) {
    for (auto fname : filenames) {
      file_status stat;
      addFile(fname, stat);
      stats.push_back(stat);
    }
  }

  void addFiles(ArrayRef<StringRef> filenames) {
    std::vector<file_status> stats;
    addFiles(filenames, stats);
  }

  void removeFile(StringRef filename) {
    SmallString<128> pathBuf;
    pathBuf = WatchedDir;
    path::append(pathBuf, filename);
    std::error_code EC = remove(pathBuf, /*IgnoreNonExisting=*/false);
    ASSERT_FALSE(EC);
  }

  void removeFiles(ArrayRef<StringRef> filenames) {
    for (auto fname : filenames) {
      removeFile(fname);
    }
  }

  /// \returns true for error.
  bool startWatching(bool waitInitialSync) {
    std::weak_ptr<DirectoryWatcherTest> weakThis = shared_from_this();
    auto receiver = [weakThis](ArrayRef<DirectoryWatcher::Event> events,
                               bool isInitial) {
      if (auto this_ = weakThis.lock())
        this_->onEvents(events, isInitial);
    };
    std::string error;
    DirWatcher = DirectoryWatcher::create(getWatchedDir(), receiver,
                                          waitInitialSync, error);
    return DirWatcher == nullptr;
  }

  void stopWatching() { DirWatcher.reset(); }

  /// \returns None if the timeout is reached before getting an event.
  Optional<EventOccurrence> getNextEvent(unsigned timeout_seconds = 5) {
    std::unique_lock<std::mutex> lck(Mutex);
    auto pred = [&]() -> bool { return !EvtOccurs.empty(); };
    bool gotEvent =
        Condition.wait_for(lck, std::chrono::seconds(timeout_seconds), pred);
    if (!gotEvent)
      return None;

    EventOccurrence occur = EvtOccurs.front();
    EvtOccurs.pop_front();
    return occur;
  }

  EventOccurrence getNextEventImmediately() {
    std::lock_guard<std::mutex> LG(Mutex);
    assert(!EvtOccurs.empty());
    EventOccurrence occur = EvtOccurs.front();
    EvtOccurs.pop_front();
    return occur;
  }

private:
  void onEvents(ArrayRef<DirectoryWatcher::Event> events, bool isInitial) {
    std::lock_guard<std::mutex> LG(Mutex);
    EvtOccurs.push_back({events, isInitial});
    Condition.notify_all();
  }
};

} // namespace

TEST(DirectoryWatcherTest, initialScan) {
  auto t = std::make_shared<DirectoryWatcherTest>();
  t->init();

  std::vector<StringRef> fnames = {"a", "b", "c"};
  std::vector<file_status> stats;
  t->addFiles(fnames, stats);

  bool err = t->startWatching(/*waitInitialSync=*/true);
  ASSERT_FALSE(err);

  auto evt = t->getNextEventImmediately();
  EXPECT_TRUE(evt.IsInitial);
  EventCollection coll1{evt.Events};
  EXPECT_TRUE(coll1.hasAdded(fnames, stats));

  StringRef additionalFname = "d";
  file_status additionalStat;
  t->addFile(additionalFname, additionalStat);
  auto evtOpt = t->getNextEvent();
  ASSERT_TRUE(evtOpt.hasValue());
  EXPECT_FALSE(evtOpt->IsInitial);
  EventCollection coll2{evtOpt->Events};
  EXPECT_TRUE(coll2.hasAdded({additionalFname}, {additionalStat}));
}

TEST(DirectoryWatcherTest, fileEvents) {
  auto t = std::make_shared<DirectoryWatcherTest>();
  t->init();

  bool err = t->startWatching(/*waitInitialSync=*/false);
  ASSERT_FALSE(err);

  auto evt = t->getNextEvent();
  ASSERT_TRUE(evt.hasValue());
  EXPECT_TRUE(evt->IsInitial);
  EXPECT_TRUE(evt->Events.empty());
  return;

  {
    std::vector<StringRef> fnames = {"a", "b"};
    std::vector<file_status> stats;
    t->addFiles(fnames, stats);

    EventCollection coll{};
    while (coll.size() < 2) {
      evt = t->getNextEvent();
      ASSERT_TRUE(evt.hasValue());
      coll.append(evt->Events);
    }
    EXPECT_TRUE(coll.hasAdded(fnames, stats));
  }
  {
    std::vector<StringRef> fnames = {"b", "c"};
    std::vector<file_status> stats;
    t->addFiles(fnames, stats);

    EventCollection coll{};
    while (coll.size() < 2) {
      evt = t->getNextEvent();
      ASSERT_TRUE(evt.hasValue());
      coll.append(evt->Events);
    }
    EXPECT_TRUE(coll.hasAdded(fnames, stats));
  }
  {
    std::vector<StringRef> fnames = {"a", "c"};
    std::vector<file_status> stats;
    t->addFiles(fnames, stats);
    t->removeFile("b");

    EventCollection coll{};
    while (coll.size() < 3) {
      evt = t->getNextEvent();
      ASSERT_TRUE(evt.hasValue());
      coll.append(evt->Events);
    }

    EXPECT_TRUE(coll.hasEvents(std::vector<StringRef>{"a", "b", "c"},
                               std::vector<DirectoryWatcher::EventKind>{
                                   DirectoryWatcher::EventKind::Added,
                                   DirectoryWatcher::EventKind::Removed,
                                   DirectoryWatcher::EventKind::Added,
                               },
                               std::vector<file_status>{
                                   stats[0],
                                   file_status{},
                                   stats[1],
                               }));
  }
  {
    std::vector<StringRef> fnames = {"a", "c"};
    t->removeFiles(fnames);

    EventCollection coll{};
    while (coll.size() < 2) {
      evt = t->getNextEvent();
      ASSERT_TRUE(evt.hasValue());
      coll.append(evt->Events);
    }
    EXPECT_TRUE(coll.hasRemoved(fnames));
  }
}
