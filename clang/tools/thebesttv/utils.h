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

struct Location {
    std::string file; // absolute path
    int line;
    int column;

    Location() : Location("", -1, -1) {}

    Location(std::string file, int line, int column)
        : file(file), line(line), column(column) {}

    Location(const FullSourceLoc &fullLoc) {
        requireTrue(fullLoc.hasManager(), "no source manager!");
        requireTrue(fullLoc.isValid(), "invalid location!");

        file = fullLoc.getFileEntry()->tryGetRealPathName();
        line = fullLoc.getLineNumber();
        column = fullLoc.getColumnNumber();
        requireTrue(!file.empty(), "empty file path!");
    }

    bool operator==(const Location &other) const {
        return file == other.file && line == other.line &&
               column == other.column;
    }
};

#endif