//===------ LangOptions.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the LangOptions interface, which holds the
//  configuration for LLVM's middle-end and back-end. It controls LLVM's code
//  generation into assembly or machine code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_FRONTEND_LANGOPTIONS_H
#define LLVM_FLANG_FRONTEND_LANGOPTIONS_H

#include <string>

namespace Fortran::frontend {

/// Bitfields of LangOptions, split out from LangOptions to ensure
/// that this large collection of bitfields is a trivial class type.
class LangOptionsBase {

public:
  enum FPModeKind {
    // Do not fuse FP ops
    FPM_Off,

    // Aggressively fuse FP ops (E.g. FMA).
    FPM_Fast,
  };

#define LANGOPT(Name, Bits, Default) unsigned Name : Bits;
#define ENUM_LANGOPT(Name, Type, Bits, Default)
#include "flang/Frontend/LangOptions.def"

protected:
#define LANGOPT(Name, Bits, Default)
#define ENUM_LANGOPT(Name, Type, Bits, Default) unsigned Name : Bits;
#include "flang/Frontend/LangOptions.def"
};

/// Tracks various options which control the dialect of Fortran that is
/// accepted. Based on clang::LangOptions
class LangOptions : public LangOptionsBase {

public:
  // Define accessors/mutators for code generation options of enumeration type.
#define LANGOPT(Name, Bits, Default)
#define ENUM_LANGOPT(Name, Type, Bits, Default)                                \
  Type get##Name() const { return static_cast<Type>(Name); }                   \
  void set##Name(Type Value) { Name = static_cast<unsigned>(Value); }
#include "flang/Frontend/LangOptions.def"

  /// Name of the IR file that contains the result of the OpenMP target
  /// host code generation.
  std::string OMPHostIRFile;

  LangOptions();
};

} // end namespace Fortran::frontend

#endif
