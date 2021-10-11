//===-- C++ code generation from NamedFunctionDescriptors -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBC_BENCHMARKS_AUTOMEMCPY_CODEGEN_H
#define LIBC_BENCHMARKS_AUTOMEMCPY_CODEGEN_H

#include "automemcpy/FunctionDescriptor.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/raw_ostream.h>
#include <vector>

namespace llvm {
namespace automemcpy {

// This function serializes the array of FunctionDescriptors as a C++ file.
void Serialize(raw_ostream &Stream, ArrayRef<FunctionDescriptor> FD);

} // namespace automemcpy
} // namespace llvm

#endif // LIBC_BENCHMARKS_AUTOMEMCPY_CODEGEN_H
