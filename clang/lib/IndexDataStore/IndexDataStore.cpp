//===--- IndexDataStore.cpp - Index data store info -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Index/IndexDataStore.h"
#include "clang/Basic/PathRemapper.h"
#include "clang/DirectoryWatcher/DirectoryWatcher.h"
#include "../lib/Index/IndexDataStoreUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::index;
using namespace clang::index::store;
using namespace llvm;

namespace {

class UnitEventHandlerData {
  mutable sys::Mutex Mtx;
  IndexDataStore::UnitEventHandler Handler;

public:
  void setHandler(IndexDataStore::UnitEventHandler handler) {
    sys::ScopedLock L(Mtx);
    Handler = std::move(handler);
  }
  IndexDataStore::UnitEventHandler getHandler() const {
    sys::ScopedLock L(Mtx);
    return Handler;
  }
};

class IndexDataStoreImpl {
  std::string FilePath;
  PathRemapper Remapper;
  std::shared_ptr<UnitEventHandlerData> TheUnitEventHandlerData;
  std::unique_ptr<DirectoryWatcher> DirWatcher;

public:
  explicit IndexDataStoreImpl(StringRef indexStorePath, PathRemapper remapper)
    : FilePath(indexStorePath), Remapper(remapper) {
    TheUnitEventHandlerData = std::make_shared<UnitEventHandlerData>();
  }

  StringRef getFilePath() const { return FilePath; }
  const PathRemapper &getPathRemapper() const {
    return Remapper;
  }
  bool foreachUnitName(bool sorted,
                       llvm::function_ref<bool(StringRef unitName)> receiver);
  void setUnitEventHandler(IndexDataStore::UnitEventHandler Handler);
  bool startEventListening(bool waitInitialSync, std::string &Error);
  void stopEventListening();
  void discardUnit(StringRef UnitName);
  void discardRecord(StringRef RecordName);
  void purgeStaleData();
};

} // anonymous namespace

bool IndexDataStoreImpl::foreachUnitName(bool sorted,
                        llvm::function_ref<bool(StringRef unitName)> receiver) {
  SmallString<128> UnitPath;
  UnitPath = FilePath;
  appendUnitSubDir(UnitPath);

  std::vector<std::string> filenames;

  std::error_code EC;
  for (auto It = sys::fs::directory_iterator(UnitPath, EC),
           End = sys::fs::directory_iterator();
       !EC && It != End; It.increment(EC)) {
    StringRef unitName = sys::path::filename(It->path());
    if (!sorted) {
      if (!receiver(unitName))
        return false;
    } else {
      filenames.push_back(std::string(unitName));
    }
  }

  if (sorted) {
    std::sort(filenames.begin(), filenames.end());
    for (auto &fname : filenames)
      if (!receiver(fname))
        return false;
  }
  return true;
}

void IndexDataStoreImpl::setUnitEventHandler(IndexDataStore::UnitEventHandler handler) {
  TheUnitEventHandlerData->setHandler(std::move(handler));
}

bool IndexDataStoreImpl::startEventListening(bool waitInitialSync, std::string &Error) {
  if (DirWatcher) {
    Error = "event listener already active";
    return true;
  }

  SmallString<128> UnitPath;
  UnitPath = FilePath;
  appendUnitSubDir(UnitPath);

  auto localUnitEventHandlerData = TheUnitEventHandlerData;
  auto OnUnitsChange = [localUnitEventHandlerData](ArrayRef<DirectoryWatcher::Event> Events, bool isInitial) {
    SmallVector<IndexDataStore::UnitEvent, 16> UnitEvents;
    UnitEvents.reserve(Events.size());
    for (const DirectoryWatcher::Event &evt : Events) {
      IndexDataStore::UnitEventKind K;
      StringRef UnitName = sys::path::filename(evt.Filename);
      switch (evt.Kind) {
      case DirectoryWatcher::Event::EventKind::Removed:
        K = IndexDataStore::UnitEventKind::Removed; break;
      case DirectoryWatcher::Event::EventKind::Modified:
        K = IndexDataStore::UnitEventKind::Modified; break;
      case DirectoryWatcher::Event::EventKind::WatchedDirRemoved:
        K = IndexDataStore::UnitEventKind::DirectoryDeleted;
        UnitName = StringRef();
        break;
      case DirectoryWatcher::Event::EventKind::WatcherGotInvalidated:
        K = IndexDataStore::UnitEventKind::Failure;
        UnitName = StringRef();
        break;
      }
      UnitEvents.push_back(IndexDataStore::UnitEvent{K, UnitName});
    }

    if (auto handler = localUnitEventHandlerData->getHandler()) {
      IndexDataStore::UnitEventNotification EventNote{isInitial, UnitEvents};
      handler(EventNote);
    }
  };

  // Create the unit path if necessary so that the directory watcher can start
  // even if the data has not been populated yet.
  if (std::error_code EC = llvm::sys::fs::create_directories(UnitPath)) {
    Error = EC.message();
    return true;
  }

  llvm::Expected<std::unique_ptr<DirectoryWatcher>> ExpectedDirWatcher =
      DirectoryWatcher::create(UnitPath.str(), OnUnitsChange, waitInitialSync);
  if (!ExpectedDirWatcher) {
      Error = llvm::toString(ExpectedDirWatcher.takeError());
      return true;
  }

  DirWatcher = std::move(ExpectedDirWatcher.get());
  return false;
}

void IndexDataStoreImpl::stopEventListening() {
  DirWatcher.reset();
}

void IndexDataStoreImpl::discardUnit(StringRef UnitName) {
  SmallString<128> UnitPath;
  UnitPath = FilePath;
  appendUnitSubDir(UnitPath);
  appendInteriorUnitPath(UnitName, UnitPath);
  sys::fs::remove(UnitPath);
}

void IndexDataStoreImpl::discardRecord(StringRef RecordName) {
  SmallString<128> RecordPath;
  RecordPath = FilePath;
  appendRecordSubDir(RecordPath);
  appendInteriorRecordPath(RecordName, RecordPath);
  sys::fs::remove(RecordPath);
}

void IndexDataStoreImpl::purgeStaleData() {
  // FIXME: Implement.
}


std::unique_ptr<IndexDataStore>
IndexDataStore::create(StringRef IndexStorePath, const PathRemapper &Remapper,
                       std::string &Error) {
  if (!sys::fs::exists(IndexStorePath)) {
    raw_string_ostream OS(Error);
    OS << "index store path does not exist: " << IndexStorePath;
    return nullptr;
  }

  return std::unique_ptr<IndexDataStore>(
    new IndexDataStore(new IndexDataStoreImpl(IndexStorePath, Remapper)));
}

#define IMPL static_cast<IndexDataStoreImpl*>(Impl)

IndexDataStore::~IndexDataStore() {
  delete IMPL;
}

StringRef IndexDataStore::getFilePath() const {
  return IMPL->getFilePath();
}

const PathRemapper & IndexDataStore::getPathRemapper() const {
  return IMPL->getPathRemapper();
}

bool IndexDataStore::foreachUnitName(bool sorted,
                     llvm::function_ref<bool(StringRef unitName)> receiver) {
  return IMPL->foreachUnitName(sorted, std::move(receiver));
}

unsigned IndexDataStore::getFormatVersion() {
  return STORE_FORMAT_VERSION;
}

void IndexDataStore::setUnitEventHandler(UnitEventHandler Handler) {
  return IMPL->setUnitEventHandler(std::move(Handler));
}

bool IndexDataStore::startEventListening(bool waitInitialSync, std::string &Error) {
  return IMPL->startEventListening(waitInitialSync, Error);
}

void IndexDataStore::stopEventListening() {
  return IMPL->stopEventListening();
}

void IndexDataStore::discardUnit(StringRef UnitName) {
  IMPL->discardUnit(UnitName);
}

void IndexDataStore::discardRecord(StringRef RecordName) {
  IMPL->discardRecord(RecordName);
}

void IndexDataStore::purgeStaleData() {
  IMPL->purgeStaleData();
}
