//===--- OpenCL.h - OpenCL enums --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines some OpenCL-specific enums.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_OPENCL_H
#define LLVM_CLANG_BASIC_OPENCL_H

namespace clang {

/// \brief Names for the OpenCL image access qualifiers (OpenCL 1.1 6.6).
enum OpenCLImageAccess {
  CLIA_read_only = 1,
  CLIA_write_only = 2,
  CLIA_read_write = 3
};

}

#endif
