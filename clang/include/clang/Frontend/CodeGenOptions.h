//===--- CodeGenOptions.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the CodeGenOptions interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_CODEGENOPTIONS_H
#define LLVM_CLANG_FRONTEND_CODEGENOPTIONS_H

#include <string>
#include <vector>

namespace clang {

/// \brief Bitfields of CodeGenOptions, split out from CodeGenOptions to ensure
/// that this large collection of bitfields is a trivial class type.
class CodeGenOptionsBase {
public:
#define CODEGENOPT(Name, Bits, Default) unsigned Name : Bits;
#define ENUM_CODEGENOPT(Name, Type, Bits, Default)
#include "clang/Frontend/CodeGenOptions.def"

protected:
#define CODEGENOPT(Name, Bits, Default)
#define ENUM_CODEGENOPT(Name, Type, Bits, Default) unsigned Name : Bits;
#include "clang/Frontend/CodeGenOptions.def"
};

/// CodeGenOptions - Track various options which control how the code
/// is optimized and passed to the backend.
class CodeGenOptions : public CodeGenOptionsBase {
public:
  enum InliningMethod {
    NoInlining,         // Perform no inlining whatsoever.
    NormalInlining,     // Use the standard function inlining pass.
    OnlyAlwaysInlining  // Only run the always inlining pass.
  };

  enum ObjCDispatchMethodKind {
    Legacy = 0,
    NonLegacy = 1,
    Mixed = 2
  };

  enum DebugInfoKind {
    NoDebugInfo,          // Don't generate debug info.
    DebugLineTablesOnly,  // Emit only debug info necessary for generating
                          // line number tables (-gline-tables-only).
    LimitedDebugInfo,     // Limit generated debug info to reduce size
                          // (-flimit-debug-info).
    FullDebugInfo         // Generate complete debug info.
  };

  enum TLSModel {
    GeneralDynamicTLSModel,
    LocalDynamicTLSModel,
    InitialExecTLSModel,
    LocalExecTLSModel
  };

  /// The code model to use (-mcmodel).
  std::string CodeModel;

  /// The filename with path we use for coverage files. The extension will be
  /// replaced.
  std::string CoverageFile;

  /// Enable additional debugging information.
  std::string DebugPass;

  /// The string to embed in debug information as the current working directory.
  std::string DebugCompilationDir;

  /// The string to embed in the debug information for the compile unit, if
  /// non-empty.
  std::string DwarfDebugFlags;

  /// The ABI to use for passing floating point arguments.
  std::string FloatABI;

  /// The float precision limit to use, if non-empty.
  std::string LimitFloatPrecision;

  /// The name of the bitcode file to link before optzns.
  std::string LinkBitcodeFile;

  /// The user provided name for the "main file", if non-empty. This is useful
  /// in situations where the input file name does not match the original input
  /// file, for example with -save-temps.
  std::string MainFileName;

  /// The name of the relocation model to use.
  std::string RelocationModel;

  /// If not an empty string, trap intrinsics are lowered to calls to this
  /// function instead of to trap instructions.
  std::string TrapFuncName;

  /// A list of command-line options to forward to the LLVM backend.
  std::vector<std::string> BackendOptions;

public:
  // Define accessors/mutators for code generation options of enumeration type.
#define CODEGENOPT(Name, Bits, Default)
#define ENUM_CODEGENOPT(Name, Type, Bits, Default) \
  Type get##Name() const { return static_cast<Type>(Name); } \
  void set##Name(Type Value) { Name = static_cast<unsigned>(Value); }
#include "clang/Frontend/CodeGenOptions.def"

  CodeGenOptions() {
#define CODEGENOPT(Name, Bits, Default) Name = Default;
#define ENUM_CODEGENOPT(Name, Type, Bits, Default) \
  set##Name(Default);
#include "clang/Frontend/CodeGenOptions.def"

    RelocationModel = "pic";
  }
};

}  // end namespace clang

#endif
