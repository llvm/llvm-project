//===--- Types.cpp - API Notes Data Types ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines data types used in the representation of API notes data.
//
//===----------------------------------------------------------------------===//
#include "clang/APINotes/Types.h"
#include "llvm/Support/raw_ostream.h"

void clang::api_notes::ObjCMethodInfo::dump(llvm::raw_ostream &os) {
    os << DesignatedInit << " " << FactoryAsInit << " " << Unavailable << " "
       << NullabilityAudited << " " << NumAdjustedNullable << " "
       << NullabilityPayload << " " << UnavailableMsg << "\n";
}

void clang::api_notes::ObjCContextInfo::dump(llvm::raw_ostream &os) {
  os << HasDefaultNullability << " " << DefaultNullability << " "
     << HasDesignatedInits << "\n";
}

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
