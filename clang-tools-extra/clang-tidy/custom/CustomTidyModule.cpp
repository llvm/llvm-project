#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "../ClangTidyOptions.h"
#include "QueryCheck.h"
#include <memory>

namespace clang::tidy {
namespace custom {

// FIXME: could be clearer to add parameter of addCheckFactories to pass
// Options?
static ClangTidyOptions const *Options = nullptr;
extern void setOptions(ClangTidyOptions const &O) { Options = &O; }

class CustomModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    if (Options == nullptr || !Options->CustomChecks.has_value() ||
        Options->CustomChecks->empty())
      return;
    for (const ClangTidyOptions::CustomCheckValue &V :
         Options->CustomChecks.value()) {
      CheckFactories.registerCheckFactory(
          // add custom- prefix to avoid conflicts with builtin checks
          "custom-" + V.Name,
          [&V](llvm::StringRef Name, ClangTidyContext *Context) {
            return std::make_unique<custom::QueryCheck>(Name, V, Context);
          });
    }
  }
};

} // namespace custom

// Register the AlteraTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<custom::CustomModule>
    X("custom-module", "Adds custom query lint checks.");

// This anchor is used to force the linker to link in the generated object file
// and thus register the AlteraModule.
volatile int CustomModuleAnchorSource = 0;

} // namespace clang::tidy
