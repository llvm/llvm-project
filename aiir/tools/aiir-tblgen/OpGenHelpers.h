//===- OpGenHelpers.h - AIIR operation generator helpers --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines helpers used in the op generators.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TOOLS_AIIRTBLGEN_OPGENHELPERS_H_
#define AIIR_TOOLS_AIIRTBLGEN_OPGENHELPERS_H_

#include "aiir/Support/LLVM.h"
#include "llvm/TableGen/Record.h"
#include <vector>

namespace aiir {
namespace tblgen {

/// Returns all the op definitions filtered by the user. The filtering is via
/// command-line option "op-include-regex" and "op-exclude-regex".
std::vector<const llvm::Record *>
getRequestedOpDefinitions(const llvm::RecordKeeper &records);

/// Checks whether `str` is a Python keyword or would shadow builtin function.
/// Regenerate using python -c"print(set(sorted(__import__('keyword').kwlist)))"
bool isPythonReserved(llvm::StringRef str);

/// Shard the op definitions into the number of shards set by "op-shard-count".
void shardOpDefinitions(
    ArrayRef<const llvm::Record *> defs,
    SmallVectorImpl<ArrayRef<const llvm::Record *>> &shardedDefs);

} // namespace tblgen
} // namespace aiir

#endif //  AIIR_TOOLS_AIIRTBLGEN_OPGENHELPERS_H_
