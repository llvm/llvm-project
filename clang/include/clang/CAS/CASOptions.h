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

/// Base class for options configuring which CAS to use. Separated for the
/// fields where we don't need special move/copy logic.
///
/// TODO: Add appropriate options once we support plugins.
class CASConfiguration {
public:
  enum CASKind {
    UnknownCAS,
    InMemoryCAS,
    OnDiskCAS,
  };

  /// Kind of CAS to use.
  CASKind getKind() const {
    return IsFrozen ? UnknownCAS : CASPath.empty() ? InMemoryCAS : OnDiskCAS;
  }

  /// Path to a persistent backing store on-disk. This is optional, although \a
  /// CASFileSystemRootID is unlikely to work without it.
  ///
  /// - "" means there is none; falls back to in-memory.
  /// - "auto" is an alias for an automatically chosen location in the user's
  ///   system cache.
  std::string CASPath;

  std::string PluginPath;
  /// Each entry is a (<option-name>, <value>) pair.
  std::vector<std::pair<std::string, std::string>> PluginOptions;

  friend bool operator==(const CASConfiguration &LHS,
                         const CASConfiguration &RHS) {
    return LHS.CASPath == RHS.CASPath && LHS.PluginPath == RHS.PluginPath &&
           LHS.PluginOptions == RHS.PluginOptions;
  }
  friend bool operator!=(const CASConfiguration &LHS,
                         const CASConfiguration &RHS) {
    return !(LHS == RHS);
  }

private:
  /// Whether the configuration has been "frozen", in order to hide the kind of
  /// CAS that's in use.
  bool IsFrozen = false;
  friend class CASOptions;
};

/// Options configuring which CAS to use. User-accessible fields should be
/// defined in CASConfiguration to enable caching a CAS instance.
///
/// CASOptions includes \a getOrCreateDatabases() for creating CAS and
/// ActionCache.
///
/// FIXME: The the caching is done here, instead of as a field in \a
/// CompilerInstance, in order to ensure that \a
/// clang::createVFSFromCompilerInvocation() uses the same CAS instance that
/// the rest of the compiler job does, without updating all callers. Probably
/// it would be better to update all callers and remove it from here.
class CASOptions : public CASConfiguration {
public:
  /// Get a CAS & ActionCache defined by the options above. Future calls will
  /// return the same instances... unless the configuration has changed, in
  /// which case new ones will be created.
  ///
  /// If \p CreateEmptyDBsOnFailure, returns empty in-memory databases on
  /// failure. Else, returns \c nullptr on failure.
  std::pair<std::shared_ptr<llvm::cas::ObjectStore>,
            std::shared_ptr<llvm::cas::ActionCache>>
  getOrCreateDatabases(DiagnosticsEngine &Diags,
                       bool CreateEmptyDBsOnFailure = false) const;

  llvm::Expected<std::pair<std::shared_ptr<llvm::cas::ObjectStore>,
                           std::shared_ptr<llvm::cas::ActionCache>>>
  getOrCreateDatabases() const;

  /// Freeze CAS Configuration. Future calls will return the same
  /// CAS instance, even if the configuration changes again later.
  ///
  /// The configuration will be wiped out to prevent it being observable or
  /// affecting the output of something that takes \a CASOptions as an input.
  /// This also "locks in" the return value of \a getOrCreateDatabases():
  /// future calls will not check if the configuration has changed.
  void freezeConfig(DiagnosticsEngine &Diags);

  /// If the configuration is not for a persistent store, it modifies it to the
  /// default on-disk CAS, otherwise this is a noop.
  void ensurePersistentCAS();

private:
  /// Initialize Cached CAS and ActionCache.
  llvm::Error initCache() const;

  struct CachedCAS {
    /// A cached CAS instance.
    std::shared_ptr<llvm::cas::ObjectStore> CAS;
    /// An ActionCache instnace.
    std::shared_ptr<llvm::cas::ActionCache> AC;

    /// Remember how the CAS was created.
    CASConfiguration Config;
  };
  mutable CachedCAS Cache;
};

} // end namespace clang

#endif // LLVM_CLANG_CAS_CASOPTIONS_H
