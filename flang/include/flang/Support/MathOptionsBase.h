//===-- include/flang/Support/MathOptionsBase.h -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Options controlling mathematical computations generated in FIR.
/// This is intended to be header-only implementation without extra
/// dependencies so that multiple components can use it to exchange
/// math configuration.
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SUPPORT_MATHOPTIONSBASE_H_
#define FORTRAN_SUPPORT_MATHOPTIONSBASE_H_

namespace Fortran::common {

class MathOptionsBase {
public:
#define ENUM_MATHOPT(Name, Type, Bits, Default) \
  Type get##Name() const { return static_cast<Type>(Name); } \
  MathOptionsBase &set##Name(Type Value) { \
    Name = static_cast<unsigned>(Value); \
    return *this; \
  }
#include "MathOptionsBase.def"

  MathOptionsBase() {
#define ENUM_MATHOPT(Name, Type, Bits, Default) set##Name(Default);
#include "MathOptionsBase.def"
  }

private:
#define ENUM_MATHOPT(Name, Type, Bits, Default) unsigned Name : Bits;
#include "MathOptionsBase.def"
};

} // namespace Fortran::common

#endif // FORTRAN_SUPPORT_MATHOPTIONSBASE_H_
