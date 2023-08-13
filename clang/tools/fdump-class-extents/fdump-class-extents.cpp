#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

namespace {
class FDumpClassExtentsConsumer : public ASTConsumer {
public:
    void HandleTranslationUnit(ASTContext &Context) override {
        SourceManager &SM = Context.getSourceManager();
        for (const auto *D : Context.getTranslationUnitDecl()->decls()) {
            if (const CXXRecordDecl *ClassDecl = dyn_cast<CXXRecordDecl>(D)) {
                // Handle class-like data structures here (class, struct, union, template class, etc.).
                SourceLocation BeginLoc = ClassDecl->getLocStart();
                SourceLocation EndLoc = ClassDecl->getLocEnd();

                const FileEntry *FE = SM.getFileEntryForID(SM.getFileID(BeginLoc));
                if (FE) {
                    llvm::outs() << "File: " << FE->getName();
                    llvm::outs() << "Class: " << ClassDecl->getQualifiedNameAsString();
                    llvm::outs() << "Line : " << SM.getLineNumber(SM.getFileID(BeginLoc), SM.getFileOffset(BeginLoc)) << "to ";
                    llvm::outs() << "Line : " << SM.getLineNumber(SM.getFileID(EndLoc), SM.getFileOffset(EndLoc)) << "\n";
                }
            }
        }
    }
};

class FDumpClassExtentsAction : public PluginASTAction {
public:
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef InFile) override {
        return std::make_unique<FDumpClassExtentsConsumer>();
    }

    bool ParseArgs(const CompilerInstance &CI, const std::vector<std::string> &args) override {
        // Add any argument parsing here if required.
        return true;
    }
};
} // namespace

// Register the plugin with Clang.
static FrontendPluginRegistry::Add<FDumpClassExtentsAction> X("fdump-class-extents", "Emit lexical extents of each class definition");