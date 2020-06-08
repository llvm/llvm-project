//===-- Lower/Image.h -- image related lowering -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Builder routines for constructing the FIR dialect of MLIR. As FIR is a
// dialect of MLIR, it makes extensive use of MLIR interfaces and MLIR's coding
// style (https://mlir.llvm.org/getting_started/DeveloperGuide/) is used in this
// module.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_IMAGE_H
#define FORTRAN_LOWER_IMAGE_H

#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/Support/BoxValue.h"

namespace Fortran {

namespace parser {
struct ChangeTeamConstruct;
struct ChangeTeamStmt;
struct EndChangeTeamStmt;
struct FormTeamStmt;
} // namespace parser

namespace evaluate {
class CoarrayRef;
} // namespace evaluate

namespace lower {
class SymMap;
namespace pft {
struct Evaluation;
} // namespace pft

//===----------------------------------------------------------------------===//
// TEAM constructs
//===----------------------------------------------------------------------===//

void genChangeTeamConstruct(AbstractConverter &, pft::Evaluation &eval,
                            const parser::ChangeTeamConstruct &);
void genChangeTeamStmt(AbstractConverter &, pft::Evaluation &eval,
                       const parser::ChangeTeamStmt &);
void genEndChangeTeamStmt(AbstractConverter &, pft::Evaluation &eval,
                          const parser::EndChangeTeamStmt &);
void genFormTeamStatement(AbstractConverter &, pft::Evaluation &eval,
                          const parser::FormTeamStmt &);

//===----------------------------------------------------------------------===//
// COARRAY expressions
//===----------------------------------------------------------------------===//

/// Coarray expression lowering helper. A coarray expression is expected to be
/// lowered into runtime support calls. For example, expressions may use a
/// message-passing runtime to access another image's data.
class CoarrayExprHelper {
public:
  explicit CoarrayExprHelper(AbstractConverter &converter, mlir::Location loc,
                             SymMap &syms)
      : converter{converter}, symMap{syms}, loc{loc} {}
  CoarrayExprHelper(const CoarrayExprHelper &) = delete;

  /// Generate the address of a co-array expression.
  ExValue genAddr(const evaluate::CoarrayRef &expr);

  /// Generate the value of a co-array expression.
  ExValue genValue(const evaluate::CoarrayRef &expr);

private:
  AbstractConverter &converter;
  SymMap &symMap;
  mlir::Location loc;
};

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_IMAGE_H
