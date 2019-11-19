//===--- OptionUtils.cpp - Utilities for command line arguments -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/OptionUtils.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "llvm/Option/ArgList.h"

using namespace clang;
using namespace llvm::opt;

template <typename IntTy>
static IntTy getLastArgIntValueImpl(const ArgList &Args, OptSpecifier Id,
                                    IntTy Default, DiagnosticsEngine *Diags) {
  IntTy Res = Default;
  if (Arg *A = Args.getLastArg(Id)) {
    if (StringRef(A->getValue()).getAsInteger(10, Res)) {
      if (Diags)
        Diags->Report(diag::err_drv_invalid_int_value)
            << A->getAsString(Args) << A->getValue();
    }
  }
  return Res;
}

namespace clang {

// Declared in clang/Frontend/Utils.h.
int getLastArgIntValue(const ArgList &Args, OptSpecifier Id, int Default,
                       DiagnosticsEngine *Diags) {
  return getLastArgIntValueImpl<int>(Args, Id, Default, Diags);
}

uint64_t getLastArgUInt64Value(const ArgList &Args, OptSpecifier Id,
                               uint64_t Default, DiagnosticsEngine *Diags) {
  return getLastArgIntValueImpl<uint64_t>(Args, Id, Default, Diags);
}

} // namespace clang
