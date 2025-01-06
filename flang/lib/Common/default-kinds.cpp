//===-- lib/Common/default-kinds.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Common/default-kinds.h"
#include "flang/Common/idioms.h"

namespace Fortran::common {

IntrinsicTypeDefaultKinds::IntrinsicTypeDefaultKinds() {
#if __x86_64__
  quadPrecisionKind_ = 10;
#endif
}

IntrinsicTypeDefaultKinds &IntrinsicTypeDefaultKinds::set_defaultIntegerKind(
    int k) {
  defaultIntegerKind_ = k;
  return *this;
}

IntrinsicTypeDefaultKinds &IntrinsicTypeDefaultKinds::set_subscriptIntegerKind(
    int k) {
  subscriptIntegerKind_ = k;
  return *this;
}

IntrinsicTypeDefaultKinds &IntrinsicTypeDefaultKinds::set_sizeIntegerKind(
    int k) {
  sizeIntegerKind_ = k;
  return *this;
}

IntrinsicTypeDefaultKinds &IntrinsicTypeDefaultKinds::set_defaultRealKind(
    int k) {
  defaultRealKind_ = k;
  return *this;
}

IntrinsicTypeDefaultKinds &IntrinsicTypeDefaultKinds::set_doublePrecisionKind(
    int k) {
  doublePrecisionKind_ = k;
  return *this;
}

IntrinsicTypeDefaultKinds &IntrinsicTypeDefaultKinds::set_quadPrecisionKind(
    int k) {
  quadPrecisionKind_ = k;
  return *this;
}

IntrinsicTypeDefaultKinds &IntrinsicTypeDefaultKinds::set_defaultCharacterKind(
    int k) {
  defaultCharacterKind_ = k;
  return *this;
}

IntrinsicTypeDefaultKinds &IntrinsicTypeDefaultKinds::set_defaultLogicalKind(
    int k) {
  defaultLogicalKind_ = k;
  return *this;
}

int IntrinsicTypeDefaultKinds::GetDefaultKind(TypeCategory category) const {
  switch (category) {
  case TypeCategory::Integer:
  case TypeCategory::Unsigned:
    return defaultIntegerKind_;
  case TypeCategory::Real:
  case TypeCategory::Complex:
    return defaultRealKind_;
  case TypeCategory::Character:
    return defaultCharacterKind_;
  case TypeCategory::Logical:
    return defaultLogicalKind_;
  default:
    CRASH_NO_CASE;
    return 0;
  }
}
} // namespace Fortran::common
