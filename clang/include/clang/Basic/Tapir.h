//===--- Tapir.h - C Language Family Language Options -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines helper functions for processing flags related to Tapir.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_TAPIR_H
#define LLVM_CLANG_BASIC_TAPIR_H

#include "llvm/Transforms/Tapir/TapirTargetIDs.h"

namespace llvm {
namespace opt {
  class ArgList;
}
}

namespace clang {

using TapirTargetID = llvm::TapirTargetID;
TapirTargetID parseTapirTarget(const llvm::opt::ArgList &Args);

} // end namespace clang

#endif
