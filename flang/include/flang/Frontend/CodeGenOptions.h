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

#ifndef FORTRAN_FRONTEND_CODEGENOPTIONS_H
#define FORTRAN_FRONTEND_CODEGENOPTIONS_H

#include "llvm/Frontend/Debug/Options.h"
#include "llvm/Frontend/Driver/CodeGenOptions.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Regex.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizerOptions.h"
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Fortran::frontend {

/// Bitfields of CodeGenOptions, split out from CodeGenOptions to ensure
/// that this large collection of bitfields is a trivial class type.
class CodeGenOptionsBase {

public:
#define CODEGENOPT(Name, Bits, Default) unsigned Name : Bits;
#define ENUM_CODEGENOPT(Name, Type, Bits, Default)
#include "flang/Frontend/CodeGenOptions.def"

protected:
#define CODEGENOPT(Name, Bits, Default)
#define ENUM_CODEGENOPT(Name, Type, Bits, Default) unsigned Name : Bits;
#include "flang/Frontend/CodeGenOptions.def"
};

/// Tracks various options which control how the code is optimized and passed
/// to the LLVM backend.
class CodeGenOptions : public CodeGenOptionsBase {

public:
  /// The paths to the pass plugins that were registered using -fpass-plugin.
  std::vector<std::string> LLVMPassPlugins;

  /// List of filenames passed in using the -fembed-offload-object option. These
  /// are offloading binaries containing device images and metadata.
  std::vector<std::string> OffloadObjects;

  /// List of filenames passed in using the -mlink-builtin-bitcode. These
  /// are bc libraries that should be linked in and internalized;
  std::vector<std::string> BuiltinBCLibs;

  /// The directory where temp files are stored if specified by -save-temps
  std::optional<std::string> SaveTempsDir;

  /// The string containing the commandline for the llvm.commandline metadata.
  std::optional<std::string> RecordCommandLine;

  /// The name of the file to which the backend should save YAML optimization
  /// records.
  std::string OptRecordFile;

  /// The regex that filters the passes that should be saved to the optimization
  /// records.
  std::string OptRecordPasses;

  /// The format used for serializing remarks (default: YAML)
  std::string OptRecordFormat;

  /// Options to add to the linker for the object file
  std::vector<std::string> DependentLibs;

  // The RemarkKind enum class and OptRemark struct are identical to what Clang
  // has
  // TODO: Share with clang instead of re-implementing here
  enum class RemarkKind {
    RK_Missing,     // Remark argument not present on the command line.
    RK_Enabled,     // Remark enabled via '-Rgroup', i.e. -Rpass, -Rpass-missed,
                    // -Rpass-analysis
    RK_Disabled,    // Remark disabled via '-Rno-group', i.e. -Rno-pass,
                    // -Rno-pass-missed, -Rno-pass-analysis.
    RK_WithPattern, // Remark pattern specified via '-Rgroup=regexp'.
  };

  /// \brief Code object version for AMDGPU.
  llvm::CodeObjectVersionKind CodeObjectVersion =
      llvm::CodeObjectVersionKind::COV_5;

  /// Optimization remark with an optional regular expression pattern.
  struct OptRemark {
    RemarkKind Kind = RemarkKind::RK_Missing;
    std::string Pattern;
    std::shared_ptr<llvm::Regex> Regex;

    /// By default, optimization remark is missing.
    OptRemark() = default;

    /// Returns true iff the optimization remark holds a valid regular
    /// expression.
    bool hasValidPattern() const { return Regex != nullptr; }

    /// Matches the given string against the regex, if there is some.
    bool patternMatches(llvm::StringRef string) const {
      return hasValidPattern() && Regex->match(string);
    }
  };

  // The OptRemark fields provided here are identical to Clang.

  /// Selected optimizations for which we should enable optimization remarks.
  /// Transformation passes whose name matches the contained (optional) regular
  /// expression (and support this feature), will emit a diagnostic whenever
  /// they perform a transformation.
  OptRemark OptimizationRemark;

  /// Selected optimizations for which we should enable missed optimization
  /// remarks. Transformation passes whose name matches the contained (optional)
  /// regular expression (and support this feature), will emit a diagnostic
  /// whenever they tried but failed to perform a transformation.
  OptRemark OptimizationRemarkMissed;

  /// Selected optimizations for which we should enable optimization analyses.
  /// Transformation passes whose name matches the contained (optional) regular
  /// expression (and support this feature), will emit a diagnostic whenever
  /// they want to explain why they decided to apply or not apply a given
  /// transformation.
  OptRemark OptimizationRemarkAnalysis;

  /// The code model to use (-mcmodel).
  std::string CodeModel;

  /// The code model-specific large data threshold to use
  /// (-mlarge-data-threshold).
  uint64_t LargeDataThreshold;

  // Define accessors/mutators for code generation options of enumeration type.
#define CODEGENOPT(Name, Bits, Default)
#define ENUM_CODEGENOPT(Name, Type, Bits, Default)                             \
  Type get##Name() const { return static_cast<Type>(Name); }                   \
  void set##Name(Type Value) { Name = static_cast<unsigned>(Value); }
#include "flang/Frontend/CodeGenOptions.def"

  CodeGenOptions();
};

std::optional<llvm::CodeModel::Model> getCodeModel(llvm::StringRef string);

} // end namespace Fortran::frontend

#endif // FORTRAN_FRONTEND_CODEGENOPTIONS_H
