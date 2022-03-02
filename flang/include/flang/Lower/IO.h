//===-- Lower/IO.h -- lower IO statements -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_IO_H
#define FORTRAN_LOWER_IO_H

namespace mlir {
class Value;
} // namespace mlir

namespace Fortran {
namespace parser {
struct ReadStmt;
struct PrintStmt;
struct WriteStmt;
} // namespace parser

namespace lower {

class AbstractConverter;

/// Generate IO call(s) for READ; return the IOSTAT code
mlir::Value genReadStatement(AbstractConverter &converter,
                             const parser::ReadStmt &stmt);

/// Generate IO call(s) for PRINT
void genPrintStatement(AbstractConverter &converter,
                       const parser::PrintStmt &stmt);

/// Generate IO call(s) for WRITE; return the IOSTAT code
mlir::Value genWriteStatement(AbstractConverter &converter,
                              const parser::WriteStmt &stmt);

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_IO_H
