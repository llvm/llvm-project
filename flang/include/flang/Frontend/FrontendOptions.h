//===- FrontendOptions.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_FLANG_FRONTEND_FRONTENDOPTIONS_H
#define LLVM_FLANG_FRONTEND_FRONTENDOPTIONS_H

#include "flang/Common/Fortran-features.h"
#include "flang/Parser/characters.h"
#include "flang/Parser/unparse.h"
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

  /// -fsyntax-only
  ParseSyntaxOnly,

  /// Emit a .o file.
  EmitObj,

  /// Parse, unparse the parse-tree and output a Fortran source file
  DebugUnparse,

  /// Parse, resolve the sybmols, unparse the parse-tree and then output a
  /// Fortran source file
  DebugUnparseWithSymbols,

  /// Parse, run semantics and then output symbols from semantics
  DebugDumpSymbols,

  /// Parse, run semantics and then output the parse tree
  DebugDumpParseTree,

  /// Dump provenance
  DebugDumpProvenance,

  /// Parse then output the number of objects in the parse tree and the overall
  /// size
  DebugMeasureParseTree,

  /// Parse, run semantics and then output the pre-FIR tree
  DebugPreFIRTree

  /// TODO: RunPreprocessor, EmitLLVM, EmitLLVMOnly,
  /// EmitCodeGenOnly, EmitAssembly, (...)
};

/// \param suffix The file extension
/// \return True if the file extension should be processed as fixed form
bool isFixedFormSuffix(llvm::StringRef suffix);

// TODO: Find a more suitable location for this. Added for compability with
// f18.cpp (this is equivalent to `asFortran` defined there).
Fortran::parser::AnalyzedObjectsAsFortran getBasicAsFortran();

/// \param suffix The file extension
/// \return True if the file extension should be processed as free form
bool isFreeFormSuffix(llvm::StringRef suffix);

enum class Language : uint8_t {
  Unknown,

  /// LLVM IR: we accept this so that we can run the optimizer on it,
  /// and compile it to assembly or object code.
  LLVM_IR,

  /// @{ Languages that the frontend can parse and compile.
  Fortran,
};

// Source file layout
enum class FortranForm {
  /// The user has not specified a form. Base the form off the file extension.
  Unknown,

  /// -ffree-form
  FixedForm,

  /// -ffixed-form
  FreeForm
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

  /// Is this input file in fixed-form format? This is simply derived from the
  /// file extension and should not be altered by consumers. For input from
  /// stdin this is never modified.
  bool isFixedForm_ = false;

public:
  FrontendInputFile() = default;
  FrontendInputFile(llvm::StringRef file, InputKind kind)
      : file_(file.str()), kind_(kind) {

    // Based on the extension, decide whether this is a fixed or free form
    // file.
    auto pathDotIndex{file.rfind(".")};
    std::string pathSuffix{file.substr(pathDotIndex + 1)};
    isFixedForm_ = isFixedFormSuffix(pathSuffix);
  }
  FrontendInputFile(const llvm::MemoryBuffer *buffer, InputKind kind)
      : buffer_(buffer), kind_(kind) {}

  InputKind kind() const { return kind_; }

  bool IsEmpty() const { return file_.empty() && buffer_ == nullptr; }
  bool IsFile() const { return !IsBuffer(); }
  bool IsBuffer() const { return buffer_ != nullptr; }
  bool IsFixedForm() const { return isFixedForm_; }

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

  // The form to process files in, if specified.
  FortranForm fortranForm_ = FortranForm::Unknown;

  // The column after which characters are ignored in fixed form lines in the
  // source file.
  int fixedFormColumns_ = 72;

  // Language features
  common::LanguageFeatureControl features_;

  // Source file encoding
  Fortran::parser::Encoding encoding_{Fortran::parser::Encoding::UTF_8};

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
