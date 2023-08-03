//===--- Types.cpp - API Notes Data Types ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines data types used in the representation of API notes data.
//
//===----------------------------------------------------------------------===//
#include "clang/APINotes/Types.h"

void clang::api_notes::ObjCMethodInfo::mergePropInfoIntoSetter(
      const ObjCPropertyInfo &pInfo) {
  // Set the type of the first argument of the the setter or check that the
  // value we have is consistent with the property.
  // TODO: Can we provide proper error handling here?
  if (auto pNullability = pInfo.getNullability()) {
    if (!NullabilityAudited) {
      addParamTypeInfo(0, *pNullability);
      assert(NumAdjustedNullable == 2);
    } else {
      assert(getParamTypeInfo(0) == *pNullability);
    }
  }
}

void clang::api_notes::ObjCMethodInfo::mergePropInfoIntoGetter(
      const ObjCPropertyInfo &pInfo) {
  // Set the return type of the getter or check that the value we have is
  // consistent with the property.
  // TODO: Can we provide proper error handling here?
  if (auto pNullability = pInfo.getNullability()) {
    if (!NullabilityAudited) {
      addReturnTypeInfo(*pNullability);
      assert(NumAdjustedNullable == 1);
    } else {
      assert(getReturnTypeInfo() == *pNullability);
    }
  }
}
