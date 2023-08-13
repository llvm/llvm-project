#ifndef FDUMP_CLASS_EXTENTS_H
#define FDUMP_CLASS_EXTENTS_H

#include "llvm-project/clang/include/clang/Frontend/FrontendPluginRegistry.h"

namespace clang {
class FDumpClassExtentsAction : public PluginASTAction {
public:
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef InFile) override;
};
} // namespace clang

#endif // FDUMP_CLASS_EXTENTS_H