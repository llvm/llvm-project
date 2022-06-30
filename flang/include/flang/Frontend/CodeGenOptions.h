//===--- CodeGenOptions.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the CodeGenOptions interface, which holds the
//  configuration for LLVM's middle-end and back-end. It controls LLVM's code
//  generation into assembly or machine code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_CODEGENOPTIONS_H
#define LLVM_CLANG_BASIC_CODEGENOPTIONS_H

#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Regex.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizerOptions.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace Fortran::frontend {

/// Bitfields of CodeGenOptions, split out from CodeGenOptions to ensure
/// that this large collection of bitfields is a trivial class type.
class CodeGenOptionsBase {

public:
#define CODEGENOPT(Name, Bits, Default) unsigned Name : Bits;
#include "flang/Frontend/CodeGenOptions.def"

protected:
#define CODEGENOPT(Name, Bits, Default)
#include "flang/Frontend/CodeGenOptions.def"
};

/// Tracks various options which control how the code is optimized and passed
/// to the LLVM backend.
class CodeGenOptions : public CodeGenOptionsBase {

public:
  CodeGenOptions();
};

} // end namespace Fortran::frontend

#endif
