//===- FrontendOptions.h ----------------------------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_FLANG_FRONTEND_FRONTENDOPTIONS_H
#define LLVM_FLANG_FRONTEND_FRONTENDOPTIONS_H

#include <cstdint>
#include <string>
namespace Fortran::frontend {

enum class Language : uint8_t {
  Unknown,

  /// LLVM IR: we accept this so that we can run the optimizer on it,
  /// and compile it to assembly or object code.
  LLVM_IR,

  ///@{ Languages that the frontend can parse and compile.
  Fortran,
  ///@}
};

/// The kind of a file that we've been handed as an input.
class InputKind {
private:
  Language lang_;

public:
  /// The input file format.
  enum Format { Source, ModuleMap, Precompiled };

  constexpr InputKind(Language l = Language::Unknown) : lang_(l) {}

  Language GetLanguage() const { return static_cast<Language>(lang_); }

  /// Is the input kind fully-unknown?
  bool IsUnknown() const { return lang_ == Language::Unknown; }
};

/// FrontendOptions - Options for controlling the behavior of the frontend.
class FrontendOptions {
public:
  /// Show the -help text.
  unsigned showHelp_ : 1;

  /// Show the -version text.
  unsigned showVersion_ : 1;

public:
  FrontendOptions() : showHelp_(false), showVersion_(false) {}
};
} // namespace Fortran::frontend

#endif // LLVM_FLANG_FRONTEND_FRONTENDOPTIONS_H
