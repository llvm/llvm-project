//===--- LowerTypes.cpp - Type translation to target-specific types -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/CodeGen/CodeGenTypes.cpp. The queries
// are adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#include "LowerTypes.h"
#include "LowerModule.h"
#include "mlir/Support/LLVM.h"

using namespace cir;

LowerTypes::LowerTypes(LowerModule &lm)
    : lm(lm), context(lm.getContext()), target(lm.getTarget()),
      CXXABI(lm.getCXXABI()),
      theABIInfo(lm.getTargetLoweringInfo().getABIInfo()),
      mlirContext(lm.getMLIRContext()), dataLayout(lm.getModule()) {}
