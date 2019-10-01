//==--- InterCheckerAPI.h ---------------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file allows introduction of checker dependencies. It contains APIs for
// inter-checker communications.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_INTERCHECKERAPI_H
#define LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_INTERCHECKERAPI_H
namespace clang {
class CheckerManager;

namespace ento {

/// Register the part of MallocChecker connected to InnerPointerChecker.
void registerInnerPointerCheckerAux(CheckerManager &Mgr);

}}
#endif /* INTERCHECKERAPI_H_ */
