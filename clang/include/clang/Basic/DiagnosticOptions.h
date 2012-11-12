//===--- DiagnosticOptions.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_DIAGNOSTICOPTIONS_H
#define LLVM_CLANG_BASIC_DIAGNOSTICOPTIONS_H

#include "llvm/ADT/IntrusiveRefCntPtr.h"

#include <string>
#include <vector>

namespace clang {

/// \brief Specifies which overload candidates to display when overload
/// resolution fails.
enum OverloadsShown {
  Ovl_All,  ///< Show all overloads.
  Ovl_Best  ///< Show just the "best" overload candidates.
};

/// DiagnosticOptions - Options for controlling the compiler diagnostics
/// engine.
class DiagnosticOptions : public llvm::RefCountedBase<DiagnosticOptions>{
public:
  enum TextDiagnosticFormat { Clang, Msvc, Vi };

  // Default values.
  enum { DefaultTabStop = 8, MaxTabStop = 100,
    DefaultMacroBacktraceLimit = 6,
    DefaultTemplateBacktraceLimit = 10,
    DefaultConstexprBacktraceLimit = 10 };

  // Define simple diagnostic options (with no accessors).
#define DIAGOPT(Name, Bits, Default) unsigned Name : Bits;
#define ENUM_DIAGOPT(Name, Type, Bits, Default)
#include "clang/Basic/DiagnosticOptions.def"

protected:
  // Define diagnostic options of enumeration type. These are private, and will
  // have accessors (below).
#define DIAGOPT(Name, Bits, Default)
#define ENUM_DIAGOPT(Name, Type, Bits, Default) unsigned Name : Bits;
#include "clang/Basic/DiagnosticOptions.def"

public:
  /// If non-empty, a file to log extended build information to, for development
  /// testing and analysis.
  std::string DumpBuildInformation;

  /// The file to log diagnostic output to.
  std::string DiagnosticLogFile;
  
  /// The file to serialize diagnostics to (non-appending).
  std::string DiagnosticSerializationFile;

  /// The list of -W... options used to alter the diagnostic mappings, with the
  /// prefixes removed.
  std::vector<std::string> Warnings;

public:
  // Define accessors/mutators for diagnostic options of enumeration type.
#define DIAGOPT(Name, Bits, Default)
#define ENUM_DIAGOPT(Name, Type, Bits, Default) \
  Type get##Name() const { return static_cast<Type>(Name); } \
  void set##Name(Type Value) { Name = static_cast<unsigned>(Value); }
#include "clang/Basic/DiagnosticOptions.def"

  DiagnosticOptions() {
#define DIAGOPT(Name, Bits, Default) Name = Default;
#define ENUM_DIAGOPT(Name, Type, Bits, Default) set##Name(Default);
#include "clang/Basic/DiagnosticOptions.def"
  }
};

typedef DiagnosticOptions::TextDiagnosticFormat TextDiagnosticFormat;

}  // end namespace clang

#endif
