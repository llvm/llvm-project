//===- raiser.h - Hotswap MC -> LLVM IR raiser entry point --------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_RAISER_H
#define HOTSWAP_TRANSPILER_RAISER_H

#include "code_object_utils.h"
#include "raise_failure.h"

#include "llvm/ADT/StringRef.h"

#include <memory>

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace COMGR::hotswap {

struct RaiseResult {
  std::unique_ptr<llvm::LLVMContext> Ctx;
  std::unique_ptr<llvm::Module> Module;
  // Structured failure description. `failure.reason == None` iff `success`.
  RaiseFailure Failure;
  bool Success = false;
};

// Raise a kernel named `KernelName` whose source ISA is `SourceISA`. `Meta`
// carries the MsgPack-derived per-kernel metadata. The scaffolding
// implementation emits a `ret void` placeholder and refuses inputs the full
// pipeline would also refuse: missing kernel descriptor, empty kernel name,
// and `SourceISA` strings that don't parse via
// `llvm::AMDGPU::parseArchAMDGCN`. The kernel-text bytes, kernel offset, and
// compilation-target ISA become real parameters once the decoder is wired
// in (subsequent commit).
RaiseResult raiseToIR(llvm::StringRef SourceISA,
                      llvm::StringRef KernelName,
                      const KernelMeta &Meta);

} // namespace COMGR::hotswap

#endif
