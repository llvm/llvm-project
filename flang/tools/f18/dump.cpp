//===-- tools/f18/dump.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file defines Dump routines available for calling from the debugger.
// Each is based on operator<< for that type. There are overloadings for
// reference and pointer, and for dumping to a provided raw_ostream or errs().

#ifdef DEBUGF18

#include "llvm/Support/raw_ostream.h"

#define DEFINE_DUMP(ns, name) \
  namespace ns { \
  class name; \
  llvm::raw_ostream &operator<<(llvm::raw_ostream &, const name &); \
  } \
  void Dump(llvm::raw_ostream &os, const ns::name &x) { os << x << '\n'; } \
  void Dump(llvm::raw_ostream &os, const ns::name *x) { \
    if (x == nullptr) \
      os << "null\n"; \
    else \
      Dump(os, *x); \
  } \
  void Dump(const ns::name &x) { Dump(llvm::errs(), x); } \
  void Dump(const ns::name *x) { Dump(llvm::errs(), *x); }

namespace Fortran {
DEFINE_DUMP(parser, Name)
DEFINE_DUMP(parser, CharBlock)
DEFINE_DUMP(semantics, Symbol)
DEFINE_DUMP(semantics, Scope)
DEFINE_DUMP(semantics, IntrinsicTypeSpec)
DEFINE_DUMP(semantics, DerivedTypeSpec)
DEFINE_DUMP(semantics, DeclTypeSpec)
} // namespace Fortran

#endif
