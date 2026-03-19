//===- CASOptions.h - Options for configuring the CAS -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines the clang::CASOptions interface.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CAS_CASOPTIONS_H
#define LLVM_CLANG_CAS_CASOPTIONS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/CAS/CASConfiguration.h"
#include "llvm/Support/Error.h"
#include <string>
#include <vector>

namespace llvm {
namespace cas {
class ActionCache;
class ObjectStore;
} // end namespace cas
} // end namespace llvm

namespace clang {

class DiagnosticsEngine;

/// Options configuring which CAS to use. User-accessible fields should be
/// defined in CASConfiguration to enable caching a CAS instance.
///
/// CASOptions includes \a createDatabases() convenience for creating CAS and
/// ActionCache and reporting diagnostics.
class CASOptions : public llvm::cas::CASConfiguration {
public:
  enum CASKind {
    InMemoryCAS,
    OnDiskCAS,
  };

  /// Kind of CAS to use.
  CASKind getKind() const { return CASPath.empty() ? InMemoryCAS : OnDiskCAS; }

  /// Get a CAS & ActionCache defined by the options above. Ignores any cached
  /// instances.
  ///
  /// If \p CreateEmptyDBsOnFailure, returns empty in-memory databases on
  /// failure. Else, returns \c nullptr on failure.
  std::pair<std::shared_ptr<llvm::cas::ObjectStore>,
            std::shared_ptr<llvm::cas::ActionCache>>
  createDatabases(DiagnosticsEngine &Diags,
                  bool CreateEmptyDBsOnFailure = false) const;

  /// If the configuration is not for a persistent store, it modifies it to the
  /// default on-disk CAS, otherwise this is a noop.
  void ensurePersistentCAS();
};

} // end namespace clang

#endif // LLVM_CLANG_CAS_CASOPTIONS_H
