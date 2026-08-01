//===- StaticLibrary.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the StaticLibrary class, which represents a static
// library of translation unit summary encodings (the SSAF analogue of an
// ar / libtool -static / lib.exe output).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_STATICLIBRARY_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_STATICLIBRARY_H

#include "clang/ScalableStaticAnalysis/Core/EntityLinker/TUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysis/Core/Model/BuildNamespace.h"
#include "llvm/TargetParser/Triple.h"
#include <memory>
#include <set>

namespace clang::ssaf {

class StaticLibraryCreateCLI;

/// Represents a static library of translation unit summary encodings.
///
/// A StaticLibrary bundles member translation units without performing
/// entity resolution, mirroring the role of ar / libtool -static / lib.exe
/// in native build pipelines. It is consumed by the EntityLinker for
/// selective inclusion when passed as a command line argument.
///
/// Static libraries are single-architecture: every member's target triple
/// must equal the library's. Multi-architecture static libraries are
/// expressed as a fat wrapper around per-architecture StaticLibrary
/// instances rather than as a single mixed-architecture library.
///
/// Members are stored as encoded TUSummaryEncoding objects: the
/// static-library tool never decodes per-entity payloads, and the linker
/// consumes them as-is during its selective inclusion pass.
class StaticLibrary {
  friend class MultiArchStaticLibrary;
  friend class SerializationFormat;
  friend class StaticLibraryCreateCLI;
  friend class TestFixture;

  /// Orders members by their TUNamespace. As a nested struct of
  /// StaticLibrary, it inherits StaticLibrary's friend access to
  /// TUSummaryEncoding's private fields.
  struct MemberByNamespace {
    bool operator()(const std::unique_ptr<TUSummaryEncoding> &A,
                    const std::unique_ptr<TUSummaryEncoding> &B) const {
      return A->TUNamespace < B->TUNamespace;
    }
  };

  // Target triple of the static library. All member TUs must share this
  // triple.
  llvm::Triple TargetTriple;

  // The namespace identifying this static library (kind=StaticLibrary).
  BuildNamespace Namespace;

  // Member translation units, ordered by their TUNamespace. Membership is
  // by namespace identity: inserting a TU whose TUNamespace already exists
  // in the set is rejected during deserialization.
  std::set<std::unique_ptr<TUSummaryEncoding>, MemberByNamespace> Members;

public:
  StaticLibrary(llvm::Triple TargetTriple, BuildNamespace Namespace)
      : TargetTriple(std::move(TargetTriple)), Namespace(std::move(Namespace)) {
  }
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_STATICLIBRARY_H
