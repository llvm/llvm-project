#pragma once

#include "utils.h"
#include "clang/AST/Mangle.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/Demangle/Demangle.h"

class GenWholeProgramCallGraphVisitor
    : public RecursiveASTVisitor<GenWholeProgramCallGraphVisitor> {

    ASTContext *Context;

    /**
     * 获取 mangling 后的函数名
     *
     * 见 https://stackoverflow.com/q/40740604
     */
    std::string getMangledName(const FunctionDecl *decl) {
        auto mangleContext = Context->createMangleContext();

        if (!mangleContext->shouldMangleDeclName(decl)) {
            return decl->getNameInfo().getName().getAsString();
        }

        std::string mangledName;
        llvm::raw_string_ostream ostream(mangledName);

        mangleContext->mangleName(decl, ostream);

        ostream.flush();

        delete mangleContext;

        return mangledName;
    };

    /**
     * 通过 Clang 的 mangling 和 llvm 的 demangling 获取函数的完整签名
     *
     * 目前不用，改用 utils.h 的 getFullSignature()
     */
    std::string getFullSignatureThroughMangling(const FunctionDecl *D) {
        return llvm::demangle(getMangledName(D));
    }

  public:
    static std::map<std::string, std::set<std::string>> callGraph;

    explicit GenWholeProgramCallGraphVisitor(ASTContext *Context)
        : Context(Context) {}

    bool VisitFunctionDecl(FunctionDecl *D) {

        FullSourceLoc FullLocation =
            D->getASTContext().getFullLoc(D->getBeginLoc());
        if (FullLocation.isInvalid() || !FullLocation.hasManager())
            return true;
        const FileEntry *fileEntry = FullLocation.getFileEntry();
        if (!fileEntry)
            return true;
        StringRef file = fileEntry->tryGetRealPathName();
        requireTrue(!file.empty());

        if (D->isThisDeclarationADefinition()) {
            std::string fullSignature = getFullSignature(D);
            // 由于 include，可能导致重复定义？
            // requireTrue(CALL_GRAPH.find(fullSignature) == CALL_GRAPH.end(),
            //             "duplicate function definition! " + fullSignature);
            if (callGraph.find(fullSignature) == callGraph.end()) {
                callGraph[fullSignature] = {};
            }

            CallGraph CG;
            CG.addToCallGraph(D);
            // CG.dump();

            CallGraphNode *N = CG.getNode(D->getCanonicalDecl());
            requireTrue(N != nullptr, "N is null!");
            for (CallGraphNode::const_iterator CI = N->begin(), CE = N->end();
                 CI != CE; ++CI) {
                FunctionDecl *callee = CI->Callee->getDecl()->getAsFunction();
                requireTrue(callee != nullptr, "callee is null!");
                callGraph[fullSignature].insert(getFullSignature(callee));
            }
        }
        return true;
    }

    bool VisitCXXRecordDecl(CXXRecordDecl *D) {
        // llvm::errs() << D->getQualifiedNameAsString() << "\n";
        return true;
    }
};

class GenWholeProgramCallGraphConsumer : public clang::ASTConsumer {
  public:
    explicit GenWholeProgramCallGraphConsumer(ASTContext *Context)
        : Visitor(Context) {}

    virtual void HandleTranslationUnit(clang::ASTContext &Context) {
        TranslationUnitDecl *TUD = Context.getTranslationUnitDecl();
        // TUD->dump();
        // CallGraph CG;
        // CG.addToCallGraph(TUD);
        // CG.dump();
        Visitor.TraverseDecl(TUD);
    }

  private:
    GenWholeProgramCallGraphVisitor Visitor;
};

class GenWholeProgramCallGraphAction : public clang::ASTFrontendAction {
  public:
    virtual std::unique_ptr<clang::ASTConsumer>
    CreateASTConsumer(clang::CompilerInstance &Compiler,
                      llvm::StringRef InFile) {
        llvm::outs() << "CreateASTConsumer in file: " << InFile << "\n";
        return std::make_unique<GenWholeProgramCallGraphConsumer>(
            &Compiler.getASTContext());
    }
};