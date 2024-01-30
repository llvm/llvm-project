#ifndef UTILS_H
#define UTILS_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/CallGraph.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>
#include <queue>
#include <set>
#include <string>

using namespace clang;
using namespace clang::tooling;
using namespace llvm;
namespace fs = std::filesystem;

struct FunctionInfo;
typedef std::map<std::string, std::set<const FunctionInfo *>> fif;

/*****************************************************************
 * Global Variables
 *****************************************************************/

extern fs::path BUILD_PATH;

/*****************************************************************
 * Utility functions
 *****************************************************************/

void requireTrue(bool condition, std::string message = "");

/**
 * 获取函数完整签名。主要用于 call graph 构造
 *
 * 包括 namespace、函数名、参数类型
 */
std::string getFullSignature(const FunctionDecl *D);

#endif