#pragma once

#include "utils.h"
#include "clang/AST/Mangle.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/Demangle/Demangle.h"

class GenWholeProgramCallGraphVisitor
    : public RecursiveASTVisitor<GenWholeProgramCallGraphVisitor> {

    ASTContext *Context;
    fs::path filePath;

    /**
     * 获取 mangling 后的函数名
     *
     * 见 https://stackoverflow.com/q/40740604
     */
    std::string getMangledName(const FunctionDecl *decl);

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
    static std::map<std::string, NamedLocation *> infoOfFunction;

    explicit GenWholeProgramCallGraphVisitor(ASTContext *Context,
                                             fs::path filePath)
        : Context(Context), filePath(filePath) {}

    bool VisitFunctionDecl(clang::FunctionDecl *D);
    bool VisitCXXRecordDecl(clang::CXXRecordDecl *D);
};

class GenWholeProgramCallGraphConsumer : public clang::ASTConsumer {
  public:
    explicit GenWholeProgramCallGraphConsumer(ASTContext *Context,
                                              fs::path filePath)
        : Visitor(Context, filePath) {}

    virtual void HandleTranslationUnit(clang::ASTContext &Context) override;

  private:
    GenWholeProgramCallGraphVisitor Visitor;
};

class GenWholeProgramCallGraphAction : public clang::ASTFrontendAction {
  public:
    virtual std::unique_ptr<clang::ASTConsumer>
    CreateASTConsumer(clang::CompilerInstance &Compiler,
                      llvm::StringRef InFile) override;
};