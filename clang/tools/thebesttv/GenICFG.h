#pragma once

#include "utils.h"
#include "clang/AST/Mangle.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/Demangle/Demangle.h"

class GenICFGVisitor : public RecursiveASTVisitor<GenICFGVisitor> {

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
    explicit GenICFGVisitor(ASTContext *Context, fs::path filePath)
        : Context(Context), filePath(filePath) {}

    bool VisitFunctionDecl(clang::FunctionDecl *D);
    bool VisitCXXRecordDecl(clang::CXXRecordDecl *D);
};

class NpeSourceVisitor : public RecursiveASTVisitor<NpeSourceVisitor> {

    ASTContext *Context;

    std::optional<typename std::set<ordered_json>::iterator>
    saveNpeSuspectedSources(const SourceRange &range);

  public:
    explicit NpeSourceVisitor(ASTContext *Context) : Context(Context) {}

    bool VisitVarDecl(VarDecl *D);
    bool VisitBinaryOperator(BinaryOperator *S);
};

class GenICFGConsumer : public clang::ASTConsumer {
  public:
    explicit GenICFGConsumer(ASTContext *Context, fs::path filePath)
        : Visitor(Context, filePath) {}

    virtual void HandleTranslationUnit(clang::ASTContext &Context) override;

  private:
    GenICFGVisitor Visitor;
};

class GenICFGAction : public clang::ASTFrontendAction {
  public:
    virtual std::unique_ptr<clang::ASTConsumer>
    CreateASTConsumer(clang::CompilerInstance &Compiler,
                      llvm::StringRef InFile) override;
};

bool updateICFGWithASTDump(const std::string &file);

/**
 * 生成全程序调用图
 */
void generateICFG(const CompilationDatabase &cb);
void generateICFGParallel(const CompilationDatabase &cb, int numThreads = 0);
