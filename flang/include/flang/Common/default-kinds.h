//===-- include/flang/Common/default-kinds.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_COMMON_DEFAULT_KINDS_H_
#define FORTRAN_COMMON_DEFAULT_KINDS_H_

#include "flang/Common/Fortran.h"
#include <cstdint>

namespace Fortran::common {

// All address calculations in generated code are 64-bit safe.
// Compile-time folding of bounds, subscripts, and lengths
// consequently uses 64-bit signed integers.  The name reflects
// this usage as a subscript into a constant array.
using ConstantSubscript = std::int64_t;

// Represent the default values of the kind parameters of the
// various intrinsic types.  Most of these can be configured by
// means of the compiler command line.
class IntrinsicTypeDefaultKinds {
public:
  IntrinsicTypeDefaultKinds();
  int subscriptIntegerKind() const { return subscriptIntegerKind_; }
  int sizeIntegerKind() const { return sizeIntegerKind_; }
  int doublePrecisionKind() const { return doublePrecisionKind_; }
  int quadPrecisionKind() const { return quadPrecisionKind_; }

  IntrinsicTypeDefaultKinds &set_defaultIntegerKind(int);
  IntrinsicTypeDefaultKinds &set_subscriptIntegerKind(int);
  IntrinsicTypeDefaultKinds &set_sizeIntegerKind(int);
  IntrinsicTypeDefaultKinds &set_defaultRealKind(int);
  IntrinsicTypeDefaultKinds &set_doublePrecisionKind(int);
  IntrinsicTypeDefaultKinds &set_quadPrecisionKind(int);
  IntrinsicTypeDefaultKinds &set_defaultCharacterKind(int);
  IntrinsicTypeDefaultKinds &set_defaultLogicalKind(int);

  int GetDefaultKind(TypeCategory) const;

private:
  // Default REAL just simply has to be IEEE-754 single precision today.
  // It occupies one numeric storage unit by definition.  The default INTEGER
  // and default LOGICAL intrinsic types also have to occupy one numeric
  // storage unit, so their kinds are also forced.  Default COMPLEX must always
  // comprise two default REAL components.
  int defaultIntegerKind_{4};
  int subscriptIntegerKind_{8};
  int sizeIntegerKind_{4}; // SIZE(), UBOUND(), &c. default KIND=
  int defaultRealKind_{defaultIntegerKind_};
  int doublePrecisionKind_{2 * defaultRealKind_};
  int quadPrecisionKind_{2 * doublePrecisionKind_};
  int defaultCharacterKind_{1};
  int defaultLogicalKind_{defaultIntegerKind_};
};
} // namespace Fortran::common
#endif // FORTRAN_COMMON_DEFAULT_KINDS_H_
