#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_MODULE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_MODULE_H

#include "LSPBinder.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/JSON.h"
#include <memory>
#include <type_traits>
#include <vector>

namespace clang {
namespace clangd {

/// A Module contributes a vertical feature to clangd.
///
/// FIXME: Extend this to support outgoing LSP calls.
///
/// The lifetime of a module is roughly:
///  - modules are created before the LSP server, in ClangdMain.cpp
///  - these modules are then passed to ClangdLSPServer and ClangdServer
///  - module hooks can be called at this point.
///    FIXME: We should make some server facilities like TUScheduler and index
///    available to those modules after ClangdServer is initalized.
///  - ClangdServer will not be destroyed until all the requests are done.
///    FIXME: Block server shutdown until all the modules are idle.
///  - modules will be destroyed after ClangdLSPServer is destroyed.
///
/// Conventionally, standard modules live in the `clangd` namespace, and other
/// exposed details live in a sub-namespace.
class Module {
public:
  virtual ~Module() = default;

  /// Called by the server to connect this module to LSP.
  /// The module should register the methods/notifications/commands it handles,
  /// and update the server capabilities to advertise them.
  ///
  /// This is only called if the module is running in ClangdLSPServer!
  /// Modules with a public interface should satisfy it without LSP bindings.
  virtual void initializeLSP(LSPBinder &Bind,
                             const llvm::json::Object &ClientCaps,
                             llvm::json::Object &ServerCaps) {}
};

/// A ModuleSet is a collection of modules installed in clangd.
///
/// Modules can be looked up by type, or used through the Module interface.
/// This allows individual modules to expose a public API.
/// For this reason, there can be only one module of each type.
///
/// ModuleSet owns the modules. It is itself owned by main, not ClangdServer.
class ModuleSet {
  std::vector<std::unique_ptr<Module>> Modules;
  llvm::DenseMap<void *, Module *> Map;

  template <typename Mod> struct ID {
    static_assert(std::is_base_of<Module, Mod>::value &&
                      std::is_final<Mod>::value,
                  "Modules must be final classes derived from clangd::Module");
    static int Key;
  };

  bool addImpl(void *Key, std::unique_ptr<Module>, const char *Source);

public:
  ModuleSet() = default;

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
    return const_cast<ModuleSet *>(this)->get<Mod>();
  }
};

template <typename Mod> int ModuleSet::ID<Mod>::Key;

} // namespace clangd
} // namespace clang
#endif
