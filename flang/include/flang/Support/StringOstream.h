//===-- CompilerInstance.h - Flang Compiler Instance ------------*- C++ -*-===//
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

#ifndef FORTRAN_SUPPORT_STRINGOSTREAM_H
#define FORTRAN_SUPPORT_STRINGOSTREAM_H

#include <llvm/Support/raw_ostream.h>

namespace Fortran::support {

/// Helper class to maintain both the an llvm::raw_string_ostream object and
/// its associated buffer.
class string_ostream : public llvm::raw_string_ostream {
private:
  std::string buf;

public:
  string_ostream() : llvm::raw_string_ostream(buf) {}
};

} // namespace Fortran::support

#endif // FORTRAN_SUPPORT_STRINGOSTREAM_H
