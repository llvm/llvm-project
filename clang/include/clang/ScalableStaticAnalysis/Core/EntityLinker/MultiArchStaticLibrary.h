//===- MultiArchStaticLibrary.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MultiArchStaticLibrary class, which represents a
// multi-architecture wrapper around per-architecture StaticLibrary
// instances.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_MULTIARCHSTATICLIBRARY_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_MULTIARCHSTATICLIBRARY_H

#include "clang/ScalableStaticAnalysis/Core/EntityLinker/StaticLibrary.h"
#include "clang/ScalableStaticAnalysis/Core/Model/BuildNamespace.h"
#include <memory>
#include <set>
#include <tuple>

namespace clang::ssaf {

/// Represents a multi-architecture static library.
///
/// A MultiArchStaticLibrary bundles per-architecture StaticLibrary members. All
/// members represent the same logical library built for different
/// architectures; the wrapper's \c Namespace identifies that shared library and
/// every member's namespace must agree on its name.
class MultiArchStaticLibrary {
  friend class SerializationFormat;
  friend class TestFixture;

  /// Orders members by their TargetTriple's canonical enum components.
  struct MemberByTargetTriple {
    static auto key(const llvm::Triple &T) {
      return std::make_tuple(T.getArch(), T.getSubArch(), T.getVendor(),
                             T.getOS(), T.getEnvironment(),
                             T.getObjectFormat());
    }
    bool operator()(const std::unique_ptr<StaticLibrary> &A,
                    const std::unique_ptr<StaticLibrary> &B) const {
      return key(A->TargetTriple) < key(B->TargetTriple);
    }
  };

  // The namespace identifying this multi-architecture library. Every member's
  // Namespace must equal this Namespace.
  BuildNamespace Namespace;

  // StaticLibrary objects ordered by TargetTriple enum components. Two
  // members with the same TargetTriple are not permitted.
  std::set<std::unique_ptr<StaticLibrary>, MemberByTargetTriple> Members;

public:
  explicit MultiArchStaticLibrary(BuildNamespace Namespace)
      : Namespace(std::move(Namespace)) {}
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_MULTIARCHSTATICLIBRARY_H
