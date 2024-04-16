#pragma once

#include "CompilationDatabase.h"
#include "utils.h"

std::string getASTDumpFile(const std::string &file);

/**
 * 获取 CompilationDatabase，并且对每条命令，修改为生成 AST dump
 */
std::unique_ptr<CompilationDatabase>
getCompilationDatabaseWithASTEmit(fs::path buildPath);

void generateASTDump(const CompilationDatabase &cb);
int generateASTDump(const CompileCommand &cmd);

std::unique_ptr<ASTUnit> getASTOfFile(const std::string &path);
