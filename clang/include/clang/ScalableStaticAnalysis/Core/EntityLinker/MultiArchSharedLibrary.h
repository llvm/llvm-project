//===- MultiArchSharedLibrary.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MultiArchSharedLibrary class, which represents a
// multi-architecture wrapper around per-architecture LUSummaryEncoding
// instances (the SSAF analogue of a Mach-O fat shared library/dylib, e.g.
// one produced by `lipo -create` over per-arch `.dylib` files).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_MULTIARCHSHAREDLIBRARY_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_MULTIARCHSHAREDLIBRARY_H

#include "clang/ScalableStaticAnalysis/Core/EntityLinker/LUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysis/Core/Model/BuildNamespace.h"
#include <memory>
#include <set>
#include <tuple>

namespace clang::ssaf {

/// Represents a multi-architecture shared library.
///
/// A MultiArchSharedLibrary bundles per-architecture LUSummaryEncoding
/// members, mirroring the role of a Mach-O fat shared library (the result
/// of `lipo -create` over per-architecture `.dylib` files). All members
/// represent the same logical shared library built for different
/// architectures; the wrapper's \c Namespace is the nested build
/// namespace identifying that shared library, and every member's
/// \c LUNamespace must equal it exactly.
class MultiArchSharedLibrary {
  friend class SerializationFormat;
  friend class TestFixture;

  /// Orders members by their TargetTriple's canonical enum components.
  /// llvm::Triple's parser maps alias spellings to the same enum values
  /// (e.g. "aarch64" and "arm64" both become Triple::aarch64), so a
  /// tuple-of-enums compare is equivalent to comparing normalized triples
  /// for identity, but without the per-call string allocation.
  struct MemberByTargetTriple {
    static auto key(const llvm::Triple &T) {
      return std::make_tuple(T.getArch(), T.getSubArch(), T.getVendor(),
                             T.getOS(), T.getEnvironment(),
                             T.getObjectFormat());
    }
    bool operator()(const std::unique_ptr<LUSummaryEncoding> &A,
                    const std::unique_ptr<LUSummaryEncoding> &B) const {
      return key(A->TargetTriple) < key(B->TargetTriple);
    }
  };

  // The nested namespace identifying this multi-architecture shared
  // library. It matches the LUNamespace of every member exactly (same
  // path, same kinds, same names): the shared library's identity is the
  // same across all its architecture slices.
  NestedBuildNamespace Namespace;

  // LUSummaryEncoding objects ordered by TargetTriple enum components.
  // Two members with the same TargetTriple are not permitted.
  std::set<std::unique_ptr<LUSummaryEncoding>, MemberByTargetTriple> Members;

public:
  explicit MultiArchSharedLibrary(NestedBuildNamespace Namespace)
      : Namespace(std::move(Namespace)) {}
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_MULTIARCHSHAREDLIBRARY_H
