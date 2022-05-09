//===--- FeatureModule.h - Plugging features into clangd ----------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_FEATUREMODULE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_FEATUREMODULE_H

#include "support/Function.h"
#include "support/Threading.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/JSON.h"
#include <memory>
#include <type_traits>
#include <vector>

namespace clang {
class CompilerInstance;
namespace clangd {
struct Diag;
class LSPBinder;
class SymbolIndex;
class ThreadsafeFS;
class TUScheduler;
class Tweak;

/// A FeatureModule contributes a vertical feature to clangd.
///
/// The lifetime of a module is roughly:
///  - feature modules are created before the LSP server, in ClangdMain.cpp
///  - these modules are then passed to ClangdLSPServer in a FeatureModuleSet
///  - initializeLSP() is called when the editor calls initialize.
//   - initialize() is then called by ClangdServer as it is constructed.
///  - module hooks can be called by the server at this point.
///    Server facilities (scheduler etc) are available.
///  - ClangdServer will not be destroyed until all the requests are done.
///    FIXME: Block server shutdown until all the modules are idle.
///  - When shutting down, ClangdServer will wait for all requests to
///    finish, call stop(), and then blockUntilIdle().
///  - feature modules will be destroyed after ClangdLSPServer is destroyed.
///
/// FeatureModules are not threadsafe in general. A module's entrypoints are:
///   - method handlers registered in initializeLSP()
///   - public methods called directly via ClangdServer.featureModule<T>()->...
///   - specific overridable "hook" methods inherited from FeatureModule
/// Unless otherwise specified, these are only called on the main thread.
///
/// Conventionally, standard feature modules live in the `clangd` namespace,
/// and other exposed details live in a sub-namespace.
class FeatureModule {
public:
  virtual ~FeatureModule() {
    /// Perform shutdown sequence on destruction in case the ClangdServer was
    /// never initialized. Usually redundant, but shutdown is idempotent.
    stop();
    blockUntilIdle(Deadline::infinity());
  }

  /// Called by the server to connect this feature module to LSP.
  /// The module should register the methods/notifications/commands it handles,
  /// and update the server capabilities to advertise them.
  ///
  /// This is only called if the module is running in ClangdLSPServer!
  /// FeatureModules with a public interface should work without LSP bindings.
  virtual void initializeLSP(LSPBinder &Bind,
                             const llvm::json::Object &ClientCaps,
                             llvm::json::Object &ServerCaps) {}

  /// Shared server facilities needed by the module to get its work done.
  struct Facilities {
    TUScheduler &Scheduler;
    const SymbolIndex *Index;
    const ThreadsafeFS &FS;
  };
  /// Called by the server to prepare this module for use.
  void initialize(const Facilities &F);

  /// Requests that the module cancel background work and go idle soon.
  /// Does not block, the caller will call blockUntilIdle() instead.
  /// After a module is stop()ed, it should not receive any more requests.
  /// Called by the server when shutting down.
  /// May be called multiple times, should be idempotent.
  virtual void stop() {}

  /// Waits until the module is idle (no background work) or a deadline expires.
  /// In general all modules should eventually go idle, though it may take a
  /// long time (e.g. background indexing).
  /// FeatureModules should go idle quickly if stop() has been called.
  /// Called by the server when shutting down, and also by tests.
  virtual bool blockUntilIdle(Deadline) { return true; }

  /// Tweaks implemented by this module. Can be called asynchronously when
  /// enumerating or applying code actions.
  virtual void contributeTweaks(std::vector<std::unique_ptr<Tweak>> &Out) {}

  /// Extension point that allows modules to observe and modify an AST build.
  /// One instance is created each time clangd produces a ParsedAST or
  /// PrecompiledPreamble. For a given instance, lifecycle methods are always
  /// called on a single thread.
  struct ASTListener {
    /// Listeners are destroyed once the AST is built.
    virtual ~ASTListener() = default;

    /// Called before every AST build, both for main file and preamble. The call
    /// happens immediately before FrontendAction::Execute(), with Preprocessor
    /// set up already and after BeginSourceFile() on main file was called.
    virtual void beforeExecute(CompilerInstance &CI) {}

    /// Called everytime a diagnostic is encountered. Modules can use this
    /// modify the final diagnostic, or store some information to surface code
    /// actions later on.
    virtual void sawDiagnostic(const clang::Diagnostic &, clangd::Diag &) {}
  };
  /// Can be called asynchronously before building an AST.
  virtual std::unique_ptr<ASTListener> astListeners() { return nullptr; }

protected:
  /// Accessors for modules to access shared server facilities they depend on.
  Facilities &facilities();
  /// The scheduler is used to run tasks on worker threads and access ASTs.
  TUScheduler &scheduler() { return facilities().Scheduler; }
  /// The index is used to get information about the whole codebase.
  const SymbolIndex *index() { return facilities().Index; }
  /// The filesystem is used to read source files on disk.
  const ThreadsafeFS &fs() { return facilities().FS; }

  /// Types of function objects that feature modules use for outgoing calls.
  /// (Bound throuh LSPBinder, made available here for convenience).
  template <typename P>
  using OutgoingNotification = llvm::unique_function<void(const P &)>;
  template <typename P, typename R>
  using OutgoingMethod = llvm::unique_function<void(const P &, Callback<R>)>;

private:
  llvm::Optional<Facilities> Fac;
};

/// A FeatureModuleSet is a collection of feature modules installed in clangd.
///
/// Modules can be looked up by type, or used via the FeatureModule interface.
/// This allows individual modules to expose a public API.
/// For this reason, there can be only one feature module of each type.
///
/// The set owns the modules. It is itself owned by main, not ClangdServer.
class FeatureModuleSet {
  std::vector<std::unique_ptr<FeatureModule>> Modules;
  llvm::DenseMap<void *, FeatureModule *> Map;

  template <typename Mod> struct ID {
    static_assert(std::is_base_of<FeatureModule, Mod>::value &&
                      std::is_final<Mod>::value,
                  "Modules must be final classes derived from clangd::Module");
    static int Key;
  };

  bool addImpl(void *Key, std::unique_ptr<FeatureModule>, const char *Source);

public:
  FeatureModuleSet() = default;

  using iterator = llvm::pointee_iterator<decltype(Modules)::iterator>;
  using const_iterator =
      llvm::pointee_iterator<decltype(Modules)::const_iterator>;
  iterator begin() { return iterator(Modules.begin()); }
  iterator end() { return iterator(Modules.end()); }
  const_iterator begin() const { return const_iterator(Modules.begin()); }
  const_iterator end() const { return const_iterator(Modules.end()); }

  template <typename Mod> bool add(std::unique_ptr<Mod> M) {
    return addImpl(&ID<Mod>::Key, std::move(M), LLVM_PRETTY_FUNCTION);
  }
  template <typename Mod> Mod *get() {
    return static_cast<Mod *>(Map.lookup(&ID<Mod>::Key));
  }
  template <typename Mod> const Mod *get() const {
    return const_cast<FeatureModuleSet *>(this)->get<Mod>();
  }
};

template <typename Mod> int FeatureModuleSet::ID<Mod>::Key;

} // namespace clangd
} // namespace clang
#endif
