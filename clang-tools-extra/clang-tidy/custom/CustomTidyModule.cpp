#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "../ClangTidyOptions.h"
#include "QueryCheck.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>
#include <memory>

namespace clang::tidy {
namespace custom {

class CustomModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {}
};

// We need to register the checks more flexibly than builtin modules. The checks
// will changed dynamically when switching to different source file.
extern void registerCustomChecks(const ClangTidyOptions &Options,
                                 ClangTidyCheckFactories &Factories) {
  static llvm::SmallSet<llvm::SmallString<32>, 8> CustomCheckNames{};
  if (!Options.CustomChecks.has_value() || Options.CustomChecks->empty())
    return;
  for (const llvm::SmallString<32> &Name : CustomCheckNames)
    Factories.eraseCheck(Name);
  for (const ClangTidyOptions::CustomCheckValue &V :
       Options.CustomChecks.value()) {
    llvm::SmallString<32> Name = llvm::StringRef{"custom-" + V.Name};
    Factories.registerCheckFactory(
        // add custom- prefix to avoid conflicts with builtin checks
        Name, [&V](llvm::StringRef Name, ClangTidyContext *Context) {
          return std::make_unique<custom::QueryCheck>(Name, V, Context);
        });
    CustomCheckNames.insert(std::move(Name));
  }
}

struct CustomChecksRegisterInitializer {
  CustomChecksRegisterInitializer() noexcept {
    RegisterCustomChecks = &custom::registerCustomChecks;
  }
};

static CustomChecksRegisterInitializer Init{};

} // namespace custom

// Register the CustomTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<custom::CustomModule>
    X("custom-module", "Adds custom query lint checks.");

// This anchor is used to force the linker to link in the generated object file
// and thus register the AlteraModule.
volatile int CustomModuleAnchorSource = 0; // NOLINT (misc-use-internal-linkage)

} // namespace clang::tidy
