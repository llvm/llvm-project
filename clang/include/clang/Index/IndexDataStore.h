//===--- IndexDataStore.h - Index data store info -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEX_INDEXDATASTORE_H
#define LLVM_CLANG_INDEX_INDEXDATASTORE_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Chrono.h"
#include <functional>
#include <memory>
#include <string>

namespace clang {
namespace index {

class IndexDataStore {
public:
  ~IndexDataStore();

  static std::unique_ptr<IndexDataStore>
    create(StringRef IndexStorePath, std::string &Error);

  StringRef getFilePath() const;
  bool foreachUnitName(bool sorted,
                       llvm::function_ref<bool(StringRef unitName)> receiver);

  static unsigned getFormatVersion();

  enum class UnitEventKind {
    Added,
    Removed,
    Modified,
    /// The directory got deleted. No more events will follow.
    DirectoryDeleted,
  };
  struct UnitEvent {
    UnitEventKind Kind;
    StringRef UnitName;
    llvm::sys::TimePoint<> ModTime;
  };
  struct UnitEventNotification {
    bool IsInitial;
    ArrayRef<UnitEvent> Events;
  };
  typedef std::function<void(UnitEventNotification)> UnitEventHandler;

  void setUnitEventHandler(UnitEventHandler Handler);
  /// \returns true if an error occurred.
  bool startEventListening(bool waitInitialSync, std::string &Error);
  void stopEventListening();

  void discardUnit(StringRef UnitName);
  void discardRecord(StringRef RecordName);

  void purgeStaleData();

private:
  IndexDataStore(void *Impl) : Impl(Impl) {}

  void *Impl; // An IndexDataStoreImpl.
};

} // namespace index
} // namespace clang

#endif
