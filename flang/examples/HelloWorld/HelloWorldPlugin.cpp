#include "flang/Frontend/FrontendActions.h"
#include "flang/Frontend/FrontendPluginRegistry.h"

__attribute__((constructor))
static void printing() {
  llvm::outs() << "      > Plugin Constructed\n";
}

using namespace Fortran::frontend;

class HelloWorldFlangPlugin : public PluginParseTreeAction
{
    void ExecuteAction() override {
      llvm::outs() << "Hello World from your new plugin (Remote plugin)\n";
    }
};

static FrontendPluginRegistry::Add<HelloWorldFlangPlugin> X("-hello-w", "Hello World Plugin example");
