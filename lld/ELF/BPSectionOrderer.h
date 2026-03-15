//===- BPSectionOrderer.h -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file uses Balanced Partitioning to order sections to improve startup
/// time and compressed size.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_BPSECTION_ORDERER_H
#define LLD_ELF_BPSECTION_ORDERER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

namespace lld::elf {
struct Ctx;
class InputSectionBase;

/// Run Balanced Partitioning to find the optimal function and data order to
/// improve startup time and compressed size.
///
/// It is important that -ffunction-sections and -fdata-sections compiler flags
/// are used to ensure functions and data are in their own sections and thus
/// can be reordered.
llvm::DenseMap<const InputSectionBase *, int>
runBalancedPartitioning(Ctx &ctx, llvm::StringRef profilePath,
                        bool forFunctionCompression, bool forDataCompression,
                        bool compressionSortStartupFunctions, bool verbose);

} // namespace lld::elf

#endif
