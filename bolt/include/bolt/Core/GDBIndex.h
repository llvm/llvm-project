//===-- bolt/Core/GDBIndex.h - GDB Index support ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file contains declaration of classes required for generation of
/// .gdb_index section.
///
//===----------------------------------------------------------------------===//

#ifndef BOLT_CORE_GDB_INDEX_H
#define BOLT_CORE_GDB_INDEX_H

#include "bolt/Core/BinaryContext.h"
#include <vector>

namespace llvm {
namespace bolt {

class GDBIndex {
public:
  /// Contains information about TU so we can write out correct entries in GDB
  /// index.
  struct GDBIndexTUEntry {
    uint64_t UnitOffset;
    uint64_t TypeHash;
    uint64_t TypeDIERelativeOffset;
  };

private:
  BinaryContext &BC;

  /// Entries for GDB Index Types CU List.
  using GDBIndexTUEntryType = std::vector<GDBIndexTUEntry>;
  GDBIndexTUEntryType GDBIndexTUEntryVector;

public:
  GDBIndex(BinaryContext &BC) : BC(BC) {}

  std::mutex GDBIndexMutex;

  /// Adds an GDBIndexTUEntry if .gdb_index section exists.
  void addGDBTypeUnitEntry(const GDBIndexTUEntry &&Entry);

  /// Rewrite .gdb_index section if present.
  void updateGdbIndexSection(const CUOffsetMap &CUMap, const uint32_t NumCUs,
                             DebugARangesSectionWriter &ARangesSectionWriter);

  /// Returns all entries needed for Types CU list.
  const GDBIndexTUEntryType &getGDBIndexTUEntryVector() const {
    return GDBIndexTUEntryVector;
  }

  /// Sorts entries in GDBIndexTUEntryVector according to the TypeHash.
  void sortGDBIndexTUEntryVector() {
    llvm::stable_sort(GDBIndexTUEntryVector, [](const GDBIndexTUEntry &LHS,
                                                const GDBIndexTUEntry &RHS) {
      return LHS.TypeHash > RHS.TypeHash;
    });
  }
};

} // namespace bolt
} // namespace llvm

#endif
