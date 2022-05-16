//===--- IndexDataStore.h - Index data store info -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEX_INDEXDATASTORE_H
#define LLVM_CLANG_INDEX_INDEXDATASTORE_H

#include "clang/Basic/LLVM.h"
#include "clang/Basic/PathRemapper.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Chrono.h"
#include <functional>
#include <memory>
#include <string>

namespace clang {
namespace index {

class IndexDataStore {
public:
  ~IndexDataStore();

  static std::unique_ptr<IndexDataStore> create(StringRef IndexStorePath,
                                                const PathRemapper &Remapper,
                                                std::string &Error);

  StringRef getFilePath() const;
  const PathRemapper &getPathRemapper() const;
  bool foreachUnitName(bool sorted,
                       llvm::function_ref<bool(StringRef unitName)> receiver);

  static unsigned getFormatVersion();

  enum class UnitEventKind {
    Removed,
    Modified,
    /// The directory got deleted. No more events will follow.
    DirectoryDeleted,
    Failure
  };
  struct UnitEvent {
    UnitEventKind Kind;
    StringRef UnitName;
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
