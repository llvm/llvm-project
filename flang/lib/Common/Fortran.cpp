//===-- lib/Common/Fortran.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Common/Fortran.h"

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

} // namespace Fortran::common
