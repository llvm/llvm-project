#pragma once

#include "CompilationDatabase.h"
#include "utils.h"

/**
 * 从 AST Dump 读取的 AST，可能为空。
 * 析构时会删除对应的 AST dump 文件。
 */
class ASTFromFile {
  private:
    std::string ASTDumpFile;
    std::shared_ptr<ASTUnit> AST;

  public:
    ASTFromFile(const std::string &file);

    ~ASTFromFile() {
        // 删除对应的 AST dump 文件
        llvm::sys::fs::remove(ASTDumpFile);
    }

    // 可能为空
    std::shared_ptr<ASTUnit> getAST() { return AST; }
};

std::string getASTDumpFile(const std::string &file);

/**
 * 获取 CompilationDatabase，并且对每条命令，修改为生成 AST dump
 */
std::unique_ptr<CompilationDatabase>
getCompilationDatabaseWithASTEmit(fs::path buildPath);

/**
 * 调用 clang 生成 AST dump
 */
int generateASTDump(const CompileCommand &cmd);

std::shared_ptr<ASTFromFile> getASTOfFile(std::string file);
