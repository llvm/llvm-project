//===-- lib/Support/default-kinds.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Support/default-kinds.h"
#include "flang/Common/idioms.h"

namespace Fortran::common {

IntrinsicTypeDefaultKinds::IntrinsicTypeDefaultKinds() {}

IntrinsicTypeDefaultKinds &IntrinsicTypeDefaultKinds::set_defaultIntegerKind(
    KindsEnum k) {
  defaultIntegerKind_ = k;
  return *this;
}

IntrinsicTypeDefaultKinds &IntrinsicTypeDefaultKinds::set_subscriptIntegerKind(
    KindsEnum k) {
  subscriptIntegerKind_ = k;
  return *this;
}

IntrinsicTypeDefaultKinds &IntrinsicTypeDefaultKinds::set_sizeIntegerKind(
    KindsEnum k) {
  sizeIntegerKind_ = k;
  return *this;
}

IntrinsicTypeDefaultKinds &IntrinsicTypeDefaultKinds::set_defaultRealKind(
    KindsEnum k) {
  defaultRealKind_ = k;
  return *this;
}

IntrinsicTypeDefaultKinds &IntrinsicTypeDefaultKinds::set_doublePrecisionKind(
    KindsEnum k) {
  doublePrecisionKind_ = k;
  return *this;
}

IntrinsicTypeDefaultKinds &IntrinsicTypeDefaultKinds::set_quadPrecisionKind(
    KindsEnum k) {
  quadPrecisionKind_ = k;
  return *this;
}

IntrinsicTypeDefaultKinds &IntrinsicTypeDefaultKinds::set_defaultCharacterKind(
    KindsEnum k) {
  defaultCharacterKind_ = k;
  return *this;
}

IntrinsicTypeDefaultKinds &IntrinsicTypeDefaultKinds::set_defaultLogicalKind(
    KindsEnum k) {
  defaultLogicalKind_ = k;
  return *this;
}

KindsEnum IntrinsicTypeDefaultKinds::GetDefaultKind(
    TypeCategory category) const {
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
    return KindsEnum::NoKind;
  }
}
} // namespace Fortran::common
