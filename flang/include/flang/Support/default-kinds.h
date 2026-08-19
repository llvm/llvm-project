//===-- include/flang/Support/default-kinds.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SUPPORT_DEFAULT_KINDS_H_
#define FORTRAN_SUPPORT_DEFAULT_KINDS_H_

#include "Fortran.h"
#include "flang/Common/type-kinds.h"
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
  KindsEnum subscriptIntegerKind() const { return subscriptIntegerKind_; }
  KindsEnum sizeIntegerKind() const { return sizeIntegerKind_; }
  KindsEnum doublePrecisionKind() const { return doublePrecisionKind_; }
  KindsEnum quadPrecisionKind() const { return quadPrecisionKind_; }

  IntrinsicTypeDefaultKinds &set_defaultIntegerKind(KindsEnum);
  IntrinsicTypeDefaultKinds &set_subscriptIntegerKind(KindsEnum);
  IntrinsicTypeDefaultKinds &set_sizeIntegerKind(KindsEnum);
  IntrinsicTypeDefaultKinds &set_defaultRealKind(KindsEnum);
  IntrinsicTypeDefaultKinds &set_doublePrecisionKind(KindsEnum);
  IntrinsicTypeDefaultKinds &set_quadPrecisionKind(KindsEnum);
  IntrinsicTypeDefaultKinds &set_defaultCharacterKind(KindsEnum);
  IntrinsicTypeDefaultKinds &set_defaultLogicalKind(KindsEnum);

  KindsEnum GetDefaultKind(TypeCategory) const;

private:
  // Default REAL just simply has to be IEEE-754 single precision today.
  // It occupies one numeric storage unit by definition.  The default INTEGER
  // and default LOGICAL intrinsic types also have to occupy one numeric
  // storage unit, so their kinds are also forced.  Default COMPLEX must always
  // comprise two default REAL components.
  KindsEnum defaultIntegerKind_{KindsEnum::Kind4};
  KindsEnum subscriptIntegerKind_{KindsEnum::Kind8};
  KindsEnum sizeIntegerKind_{
      KindsEnum::Kind4}; // SIZE(), UBOUND(), &c. default KIND=
  KindsEnum defaultRealKind_{defaultIntegerKind_};
  KindsEnum doublePrecisionKind_{
      static_cast<KindsEnum>(2 * static_cast<int>(defaultRealKind_))};
  KindsEnum quadPrecisionKind_{
      static_cast<KindsEnum>(2 * static_cast<int>(doublePrecisionKind_))};
  KindsEnum defaultCharacterKind_{KindsEnum::Kind1};
  KindsEnum defaultLogicalKind_{defaultIntegerKind_};
};
} // namespace Fortran::common
#endif // FORTRAN_SUPPORT_DEFAULT_KINDS_H_
