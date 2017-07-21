//===-- DiagnosticsYaml.h -- Serialiazation for Diagnosticss ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file defines the structure of a YAML document for serializing
/// diagnostics.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_DIAGNOSTICSYAML_H
#define LLVM_CLANG_TOOLING_DIAGNOSTICSYAML_H

#include "clang/Tooling/Core/Diagnostic.h"
#include "clang/Tooling/ReplacementsYaml.h"
#include "llvm/Support/YAMLTraits.h"
#include <string>

LLVM_YAML_IS_SEQUENCE_VECTOR(clang::tooling::Diagnostic)

namespace llvm {
namespace yaml {

template <> struct MappingTraits<clang::tooling::Diagnostic> {
  /// \brief Helper to (de)serialize a Diagnostic since we don't have direct
  /// access to its data members.
  class NormalizedDiagnostic {
  public:
    NormalizedDiagnostic(const IO &)
        : DiagLevel(clang::tooling::Diagnostic::Level::Warning) {}

    NormalizedDiagnostic(const IO &, const clang::tooling::Diagnostic &D)
        : DiagnosticName(D.DiagnosticName), Message(D.Message), Fix(D.Fix),
          Notes(D.Notes), DiagLevel(D.DiagLevel),
          BuildDirectory(D.BuildDirectory) {}

    clang::tooling::Diagnostic denormalize(const IO &) {
      return clang::tooling::Diagnostic(DiagnosticName, Message, Fix, Notes,
                                        DiagLevel, BuildDirectory);
    }

    std::string DiagnosticName;
    clang::tooling::DiagnosticMessage Message;
    llvm::StringMap<clang::tooling::Replacements> Fix;
    SmallVector<clang::tooling::DiagnosticMessage, 1> Notes;
    clang::tooling::Diagnostic::Level DiagLevel;
    std::string BuildDirectory;
  };

  static void mapping(IO &Io, clang::tooling::Diagnostic &D) {
    MappingNormalization<NormalizedDiagnostic, clang::tooling::Diagnostic> Keys(
        Io, D);
    Io.mapRequired("DiagnosticName", Keys->DiagnosticName);
    Io.mapRequired("Message", Keys->Message.Message);
    Io.mapRequired("FileOffset", Keys->Message.FileOffset);
    Io.mapRequired("FilePath", Keys->Message.FilePath);

    // FIXME: Export properly all the different fields.

    std::vector<clang::tooling::Replacement> Fixes;
    for (auto &Replacements : Keys->Fix) {
      for (auto &Replacement : Replacements.second) {
        Fixes.push_back(Replacement);
      }
    }
    Io.mapRequired("Replacements", Fixes);
    for (auto &Fix : Fixes) {
      llvm::Error Err = Keys->Fix[Fix.getFilePath()].add(Fix);
      if (Err) {
        // FIXME: Implement better conflict handling.
        llvm::errs() << "Fix conflicts with existing fix: "
                     << llvm::toString(std::move(Err)) << "\n";
      }
    }
  }
};

/// \brief Specialized MappingTraits to describe how a
/// TranslationUnitDiagnostics is (de)serialized.
template <> struct MappingTraits<clang::tooling::TranslationUnitDiagnostics> {
  static void mapping(IO &Io, clang::tooling::TranslationUnitDiagnostics &Doc) {
    Io.mapRequired("MainSourceFile", Doc.MainSourceFile);
    Io.mapRequired("Diagnostics", Doc.Diagnostics);
  }
};
} // end namespace yaml
} // end namespace llvm

#endif // LLVM_CLANG_TOOLING_DIAGNOSTICSYAML_H
