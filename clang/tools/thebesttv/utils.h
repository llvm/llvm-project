#ifndef UTILS_H
#define UTILS_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/CallGraph.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <filesystem>
#include <queue>
#include <set>
#include <string>

#include "ICFG.h"

using namespace clang;
using namespace clang::tooling;
using namespace llvm;
namespace fs = std::filesystem;

struct FunctionInfo;
typedef std::map<std::string, std::set<const FunctionInfo *>> fif;

/*****************************************************************
 * Global Variables
 *****************************************************************/

struct GlobalStat;
extern GlobalStat Global;

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

struct Location {
    std::string file; // absolute path
    int line;
    int column;

    Location();
    Location(std::string file, int line, int column);

    bool operator==(const Location &other) const;
};

struct NamedLocation : public Location {
    std::string name;

    NamedLocation();
    NamedLocation(std::string file, int line, int column, std::string name);

    bool operator==(const NamedLocation &other) const;
};

struct GlobalStat {
    std::unique_ptr<CompilationDatabase> cb;

    fs::path buildPath;

    std::map<std::string, std::set<std::string>> callGraph;

    int functionCnt;
    std::map<std::string, int> idOfFunction;
    std::vector<NamedLocation *> functionLocations;

    int getIdOfFunction(const std::string &signature) {
        auto it = idOfFunction.find(signature);
        if (it == idOfFunction.end()) {
            return -1;
        }
        return it->second;
    }

    ICFG icfg;
};

#endif