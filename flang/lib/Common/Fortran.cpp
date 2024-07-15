//===-- lib/Common/Fortran.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Common/Fortran.h"
#include "flang/Common/Fortran-features.h"

namespace Fortran::common {

const char *AsFortran(NumericOperator opr) {
  switch (opr) {
    SWITCH_COVERS_ALL_CASES
  case NumericOperator::Power:
    return "**";
  case NumericOperator::Multiply:
    return "*";
  case NumericOperator::Divide:
    return "/";
  case NumericOperator::Add:
    return "+";
  case NumericOperator::Subtract:
    return "-";
  }
}

const char *AsFortran(LogicalOperator opr) {
  switch (opr) {
    SWITCH_COVERS_ALL_CASES
  case LogicalOperator::And:
    return ".and.";
  case LogicalOperator::Or:
    return ".or.";
  case LogicalOperator::Eqv:
    return ".eqv.";
  case LogicalOperator::Neqv:
    return ".neqv.";
  case LogicalOperator::Not:
    return ".not.";
  }
}

const char *AsFortran(RelationalOperator opr) {
  switch (opr) {
    SWITCH_COVERS_ALL_CASES
  case RelationalOperator::LT:
    return "<";
  case RelationalOperator::LE:
    return "<=";
  case RelationalOperator::EQ:
    return "==";
  case RelationalOperator::NE:
    return "/=";
  case RelationalOperator::GE:
    return ">=";
  case RelationalOperator::GT:
    return ">";
  }
}

const char *AsFortran(DefinedIo x) {
  switch (x) {
    SWITCH_COVERS_ALL_CASES
  case DefinedIo::ReadFormatted:
    return "read(formatted)";
  case DefinedIo::ReadUnformatted:
    return "read(unformatted)";
  case DefinedIo::WriteFormatted:
    return "write(formatted)";
  case DefinedIo::WriteUnformatted:
    return "write(unformatted)";
  }
}

std::string AsFortran(IgnoreTKRSet tkr) {
  std::string result;
  if (tkr.test(IgnoreTKR::Type)) {
    result += 'T';
  }
  if (tkr.test(IgnoreTKR::Kind)) {
    result += 'K';
  }
  if (tkr.test(IgnoreTKR::Rank)) {
    result += 'R';
  }
  if (tkr.test(IgnoreTKR::Device)) {
    result += 'D';
  }
  if (tkr.test(IgnoreTKR::Managed)) {
    result += 'M';
  }
  if (tkr.test(IgnoreTKR::Contiguous)) {
    result += 'C';
  }
  return result;
}

/// Check compatibilty of CUDA attribute.
/// When `allowUnifiedMatchingRule` is enabled, argument `x` represents the
/// dummy argument attribute while `y` represents the actual argument attribute.
bool AreCompatibleCUDADataAttrs(std::optional<CUDADataAttr> x,
    std::optional<CUDADataAttr> y, IgnoreTKRSet ignoreTKR,
    bool allowUnifiedMatchingRule, const LanguageFeatureControl *features) {
  bool isCudaManaged{features
          ? features->IsEnabled(common::LanguageFeature::CudaManaged)
          : false};
  bool isCudaUnified{features
          ? features->IsEnabled(common::LanguageFeature::CudaUnified)
          : false};
  if (!x && !y) {
    return true;
  } else if (x && y && *x == *y) {
    return true;
  } else if ((!x && y && *y == CUDADataAttr::Pinned) ||
      (x && *x == CUDADataAttr::Pinned && !y)) {
    return true;
  } else if (ignoreTKR.test(IgnoreTKR::Device) &&
      x.value_or(CUDADataAttr::Device) == CUDADataAttr::Device &&
      y.value_or(CUDADataAttr::Device) == CUDADataAttr::Device) {
    return true;
  } else if (ignoreTKR.test(IgnoreTKR::Managed) &&
      x.value_or(CUDADataAttr::Managed) == CUDADataAttr::Managed &&
      y.value_or(CUDADataAttr::Managed) == CUDADataAttr::Managed) {
    return true;
  } else if (allowUnifiedMatchingRule) {
    if (!x) { // Dummy argument has no attribute -> host
      if ((y && (*y == CUDADataAttr::Managed || *y == CUDADataAttr::Unified)) ||
          (!y && (isCudaUnified || isCudaManaged))) {
        return true;
      }
    } else {
      if (*x == CUDADataAttr::Device) {
        if ((y &&
                (*y == CUDADataAttr::Managed || *y == CUDADataAttr::Unified)) ||
            (!y && (isCudaUnified || isCudaManaged))) {
          return true;
        }
      } else if (*x == CUDADataAttr::Managed) {
        if ((y && *y == CUDADataAttr::Unified) ||
            (!y && (isCudaUnified || isCudaManaged))) {
          return true;
        }
      } else if (*x == CUDADataAttr::Unified) {
        if ((y && *y == CUDADataAttr::Managed) ||
            (!y && (isCudaUnified || isCudaManaged))) {
          return true;
        }
      }
    }
    return false;
  } else {
    return false;
  }
}

} // namespace Fortran::common
