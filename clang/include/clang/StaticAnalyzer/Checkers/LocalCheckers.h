//==- LocalCheckers.h - Intra-Procedural+Flow-Sensitive Checkers -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the interface to call a set of intra-procedural (local)
//  checkers that use flow/path-sensitive analyses to find bugs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CHECKERS_LOCALCHECKERS_H
#define LLVM_CLANG_STATICANALYZER_CHECKERS_LOCALCHECKERS_H

namespace clang {
namespace ento {

class ExprEngine;

void RegisterCallInliner(ExprEngine &Eng);

} // end namespace ento
} // end namespace clang

#endif
