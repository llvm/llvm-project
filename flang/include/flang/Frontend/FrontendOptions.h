//===- FrontendOptions.h ----------------------------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_FLANG_FRONTEND_FRONTENDOPTIONS_H
#define LLVM_FLANG_FRONTEND_FRONTENDOPTIONS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cstdint>
#include <string>

namespace Fortran::frontend {

enum ActionKind {
  InvalidAction = 0,

  /// -test-io mode
  InputOutputTest,

  /// -E mode.
  PrintPreprocessedInput,
  /// TODO: RunPreprocessor, ParserSyntaxOnly, EmitLLVM, EmitLLVMOnly,
  /// EmitCodeGenOnly, EmitAssembly, (...)
};

inline const char *GetActionKindName(const ActionKind ak) {
  switch (ak) {
  case InputOutputTest:
    return "InputOutputTest";
  case PrintPreprocessedInput:
    return "PrintPreprocessedInput";
  default:
    return "<unknown ActionKind>";
    // TODO:
    // case RunPreprocessor:
    // case ParserSyntaxOnly:
    // case EmitLLVM:
    // case EmitLLVMOnly:
    // case EmitCodeGenOnly:
    // (...)
  }
}

enum class Language : uint8_t {
  Unknown,

  /// LLVM IR: we accept this so that we can run the optimizer on it,
  /// and compile it to assembly or object code.
  LLVM_IR,

  /// @{ Languages that the frontend can parse and compile.
  Fortran,
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

/// An input file for the front end.
class FrontendInputFile {
  /// The file name, or "-" to read from standard input.
  std::string file_;

  /// The input, if it comes from a buffer rather than a file. This object
  /// does not own the buffer, and the caller is responsible for ensuring
  /// that it outlives any users.
  const llvm::MemoryBuffer *buffer_ = nullptr;

  /// The kind of input, atm it contains language
  InputKind kind_;

public:
  FrontendInputFile() = default;
  FrontendInputFile(llvm::StringRef file, InputKind kind)
      : file_(file.str()), kind_(kind) {}
  FrontendInputFile(const llvm::MemoryBuffer *buffer, InputKind kind)
      : buffer_(buffer), kind_(kind) {}

  InputKind kind() const { return kind_; }

  bool IsEmpty() const { return file_.empty() && buffer_ == nullptr; }
  bool IsFile() const { return !IsBuffer(); }
  bool IsBuffer() const { return buffer_ != nullptr; }

  llvm::StringRef file() const {
    assert(IsFile());
    return file_;
  }

  const llvm::MemoryBuffer *buffer() const {
    assert(IsBuffer() && "Requested buffer_, but it is empty!");
    return buffer_;
  }
};

/// FrontendOptions - Options for controlling the behavior of the frontend.
class FrontendOptions {
public:
  /// Show the -help text.
  unsigned showHelp_ : 1;

  /// Show the -version text.
  unsigned showVersion_ : 1;

  /// The input files and their types.
  std::vector<FrontendInputFile> inputs_;

  /// The output file, if any.
  std::string outputFile_;

  /// The frontend action to perform.
  frontend::ActionKind programAction_;

public:
  FrontendOptions() : showHelp_(false), showVersion_(false) {}

  // Return the appropriate input kind for a file extension. For example,
  /// "*.f" would return Language::Fortran.
  ///
  /// \return The input kind for the extension, or Language::Unknown if the
  /// extension is not recognized.
  static InputKind GetInputKindForExtension(llvm::StringRef extension);
};
} // namespace Fortran::frontend

#endif // LLVM_FLANG_FRONTEND_FRONTENDOPTIONS_H
