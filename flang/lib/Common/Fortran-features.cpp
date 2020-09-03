//===-- lib/Common/Fortran-features.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Common/Fortran-features.h"
#include "flang/Common/Fortran.h"
#include "flang/Common/idioms.h"

namespace Fortran::common {

std::vector<const char *> LanguageFeatureControl::GetNames(
    LogicalOperator opr) const {
  std::vector<const char *> result;
  result.push_back(AsFortran(opr));
  if (opr == LogicalOperator::Neqv && IsEnabled(LanguageFeature::XOROperator)) {
    result.push_back(".xor.");
  }
  if (IsEnabled(LanguageFeature::LogicalAbbreviations)) {
    switch (opr) {
      SWITCH_COVERS_ALL_CASES
    case LogicalOperator::And:
      result.push_back(".a.");
      break;
    case LogicalOperator::Or:
      result.push_back(".o.");
      break;
    case LogicalOperator::Not:
      result.push_back(".n.");
      break;
    case LogicalOperator::Neqv:
      if (IsEnabled(LanguageFeature::XOROperator)) {
        result.push_back(".x.");
      }
      break;
    case LogicalOperator::Eqv:
      break;
    }
  }
  return result;
}

std::vector<const char *> LanguageFeatureControl::GetNames(
    RelationalOperator opr) const {
  switch (opr) {
    SWITCH_COVERS_ALL_CASES
  case RelationalOperator::LT:
    return {".lt.", "<"};
  case RelationalOperator::LE:
    return {".le.", "<="};
  case RelationalOperator::EQ:
    return {".eq.", "=="};
  case RelationalOperator::GE:
    return {".ge.", ">="};
  case RelationalOperator::GT:
    return {".gt.", ">"};
  case RelationalOperator::NE:
    if (IsEnabled(LanguageFeature::AlternativeNE)) {
      return {".ne.", "/=", "<>"};
    } else {
      return {".ne.", "/="};
    }
  }
}

} // namespace Fortran::common
