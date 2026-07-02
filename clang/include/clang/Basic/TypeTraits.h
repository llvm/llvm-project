//===--- TypeTraits.h - C++ Type Traits Support Enumerations ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines enumerations for the type traits support.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_TYPETRAITS_H
#define LLVM_CLANG_BASIC_TYPETRAITS_H

#include "llvm/Support/Compiler.h"

namespace clang {
/// Names for traits that operate specifically on types.
// enum TypeTrait {
//   UTT_ ...
//   UTT_Last == last UTT_XX in the enum.
//   BTT_ ...
//   BTT_Last == last BTT_XX in the enum.
//   TT_ ...
//   TT_Last == last TT_XX in the enum.
// };

/// Names for the array type traits.
// enum ArrayTypeTrait {
//   ATT_ ...
//   ATT_Last == last ATT_XX in the enum.
// };

/// Names for the "expression or type" traits.
// enum UnaryExprOrTypeTrait {
//   UETT_ ...
//   UETT_Last == last UETT_XX in the enum.
// };
#define EMIT_ENUMS
#include "clang/Basic/Traits.inc"

/// Return the internal name of type trait \p T. Never null.
const char *getTraitName(TypeTrait T) LLVM_READONLY;
const char *getTraitName(ArrayTypeTrait T) LLVM_READONLY;
const char *getTraitName(UnaryExprOrTypeTrait T) LLVM_READONLY;

/// Return the spelling of the type trait \p TT. Never null.
const char *getTraitSpelling(TypeTrait T) LLVM_READONLY;
const char *getTraitSpelling(ArrayTypeTrait T) LLVM_READONLY;
const char *getTraitSpelling(UnaryExprOrTypeTrait T) LLVM_READONLY;

/// Return the arity of the type trait \p T.
unsigned getTypeTraitArity(TypeTrait T) LLVM_READONLY;

} // namespace clang

#endif
