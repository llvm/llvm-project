//===- TypeMerger.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_TYPEMERGER_H
#define LLD_COFF_TYPEMERGER_H

#include "Config.h"
#include "llvm/DebugInfo/CodeView/GlobalTypeTableBuilder.h"
#include "llvm/DebugInfo/CodeView/MergingTypeTableBuilder.h"
#include "llvm/Support/Allocator.h"

namespace lld {
namespace coff {

class TypeMerger {
public:
  TypeMerger(llvm::BumpPtrAllocator &Alloc)
      : TypeTable(Alloc), IDTable(Alloc), GlobalTypeTable(Alloc),
        GlobalIDTable(Alloc) {}

  /// Get the type table or the global type table if /DEBUG:GHASH is enabled.
  inline llvm::codeview::TypeCollection &getTypeTable() {
    if (Config->DebugGHashes)
      return GlobalTypeTable;
    return TypeTable;
  }

  /// Get the ID table or the global ID table if /DEBUG:GHASH is enabled.
  inline llvm::codeview::TypeCollection &getIDTable() {
    if (Config->DebugGHashes)
      return GlobalIDTable;
    return IDTable;
  }

  /// Type records that will go into the PDB TPI stream.
  llvm::codeview::MergingTypeTableBuilder TypeTable;

  /// Item records that will go into the PDB IPI stream.
  llvm::codeview::MergingTypeTableBuilder IDTable;

  /// Type records that will go into the PDB TPI stream (for /DEBUG:GHASH)
  llvm::codeview::GlobalTypeTableBuilder GlobalTypeTable;

  /// Item records that will go into the PDB IPI stream (for /DEBUG:GHASH)
  llvm::codeview::GlobalTypeTableBuilder GlobalIDTable;
};

/// Map from type index and item index in a type server PDB to the
/// corresponding index in the destination PDB.
struct CVIndexMap {
  llvm::SmallVector<llvm::codeview::TypeIndex, 0> TPIMap;
  llvm::SmallVector<llvm::codeview::TypeIndex, 0> IPIMap;
  bool IsTypeServerMap = false;
  bool IsPrecompiledTypeMap = false;
};

} // namespace coff
} // namespace lld

#endif