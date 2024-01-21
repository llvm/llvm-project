#ifndef UTILS_H
#define UTILS_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Stmt.h"
#include "clang/Analysis/CFG.h"
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

/*****************************************************************
 * Global Variables
 *****************************************************************/

extern fs::path BUILD_PATH;

struct FunctionInfo;
extern std::map<std::string, std::set<const FunctionInfo *>> functionsInFile;

/*****************************************************************
 * Utility functions
 *****************************************************************/

void requireTrue(bool condition, std::string message = "");

#endif