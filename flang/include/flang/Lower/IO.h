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
struct BackspaceStmt;
struct CloseStmt;
struct EndfileStmt;
struct FlushStmt;
struct InquireStmt;
struct OpenStmt;
struct ReadStmt;
struct RewindStmt;
struct PrintStmt;
struct WaitStmt;
struct WriteStmt;
} // namespace parser

namespace lower {

class AbstractConverter;

/// Generate IO call(s) for BACKSPACE; return the IOSTAT code
mlir::Value genBackspaceStatement(AbstractConverter &,
                                  const parser::BackspaceStmt &);

/// Generate IO call(s) for CLOSE; return the IOSTAT code
mlir::Value genCloseStatement(AbstractConverter &, const parser::CloseStmt &);

/// Generate IO call(s) for ENDFILE; return the IOSTAT code
mlir::Value genEndfileStatement(AbstractConverter &,
                                const parser::EndfileStmt &);

/// Generate IO call(s) for FLUSH; return the IOSTAT code
mlir::Value genFlushStatement(AbstractConverter &, const parser::FlushStmt &);

/// Generate IO call(s) for INQUIRE; return the IOSTAT code
mlir::Value genInquireStatement(AbstractConverter &,
                                const parser::InquireStmt &);

/// Generate IO call(s) for READ; return the IOSTAT code
mlir::Value genReadStatement(AbstractConverter &converter,
                             const parser::ReadStmt &stmt);

/// Generate IO call(s) for OPEN; return the IOSTAT code
mlir::Value genOpenStatement(AbstractConverter &, const parser::OpenStmt &);

/// Generate IO call(s) for PRINT
void genPrintStatement(AbstractConverter &converter,
                       const parser::PrintStmt &stmt);

/// Generate IO call(s) for REWIND; return the IOSTAT code
mlir::Value genRewindStatement(AbstractConverter &, const parser::RewindStmt &);

/// Generate IO call(s) for WAIT; return the IOSTAT code
mlir::Value genWaitStatement(AbstractConverter &, const parser::WaitStmt &);

/// Generate IO call(s) for WRITE; return the IOSTAT code
mlir::Value genWriteStatement(AbstractConverter &converter,
                              const parser::WriteStmt &stmt);

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_IO_H
