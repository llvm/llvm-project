#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_MODULE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_MODULE_H

#include "llvm/ADT/StringRef.h"
#include <memory>
#include <vector>

namespace clang {
namespace clangd {

/// A Module contributes a vertical feature to clangd.
///
/// FIXME: Extend this with LSP bindings to support reading/updating
/// capabilities and implementing LSP endpoints.
///
/// The lifetime of a module is roughly:
///  - modules are created before the LSP server, in ClangdMain.cpp
///  - these modules are then passed to ClangdLSPServer and ClangdServer
///    FIXME: LSP bindings should be registered at ClangdLSPServer
///    initialization.
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
};

class ModuleSet {
  std::vector<std::unique_ptr<Module>> Modules;

public:
  explicit ModuleSet(std::vector<std::unique_ptr<Module>> Modules)
      : Modules(std::move(Modules)) {}

  using iterator = llvm::pointee_iterator<decltype(Modules)::iterator>;
  using const_iterator =
      llvm::pointee_iterator<decltype(Modules)::const_iterator>;
  iterator begin() { return iterator(Modules.begin()); }
  iterator end() { return iterator(Modules.end()); }
  const_iterator begin() const { return const_iterator(Modules.begin()); }
  const_iterator end() const { return const_iterator(Modules.end()); }
};
} // namespace clangd
} // namespace clang
#endif
