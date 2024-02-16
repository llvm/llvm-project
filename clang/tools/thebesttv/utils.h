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
#include <memory>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <vector>

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

    Location() : Location("", -1, -1) {}

    Location(const std::string &file, int line, int column)
        : file(file), line(line), column(column) {}

    bool operator==(const Location &other) const {
        return file == other.file && line == other.line &&
               column == other.column;
    }

    static std::unique_ptr<Location>
    fromSourceLocation(const ASTContext &Context, const SourceLocation &loc);
};

struct NamedLocation : public Location {
    std::string name;

    NamedLocation() : NamedLocation("", -1, -1, "") {}

    NamedLocation(const std::string &file, int line, int column,
                  const std::string &name)
        : Location(file, line, column), name(name) {}

    NamedLocation(const Location &loc, const std::string &name)
        : Location(loc), name(name) {}

    bool operator==(const NamedLocation &other) const {
        return Location::operator==(other) && name == other.name;
    }
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