#include "Module.h"

namespace clang {
namespace clangd {

bool ModuleSet::addImpl(void *Key, std::unique_ptr<Module> M,
                        const char *Source) {
  if (!Map.try_emplace(Key, M.get()).second) {
    // Source should (usually) include the name of the concrete module type.
    elog("Tried to register duplicate modules via {0}", Source);
    return false;
  }
  Modules.push_back(std::move(M));
  return true;
}

} // namespace clangd
} // namespace clang
