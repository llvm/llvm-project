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
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <vector>

#include "ICFG.h"

#include "lib/json.hpp"

// doesn't seem useful
// #include "spdlog/fmt/ostr.h"
#include "spdlog/spdlog.h"

using json = nlohmann::json;
using ordered_json = nlohmann::ordered_json;

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

extern spdlog::logger &logger;

/*****************************************************************
 * fmt::formatter for std::filesystem::path
 *****************************************************************/

template <>
struct fmt::formatter<std::filesystem::path> : fmt::formatter<std::string> {
    auto format(const std::filesystem::path &p, format_context &ctx) const
        -> decltype(ctx.out()) {
        return formatter<std::string>::format(p.string(), ctx);
    }
};

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

void dumpSourceLocation(const std::string &msg, const ASTContext &Context,
                        const SourceLocation &loc);

void dumpStmt(const ASTContext &Context, const Stmt *S);

struct Location {
    std::string file; // absolute path
    int line;
    int column;

    Location() : Location("", -1, -1) {}

    Location(const json &j)
        : file(j["file"]), line(j["line"]), column(j["column"]) {}

    Location(const std::string &file, int line, int column)
        : file(file), line(line), column(column) {}

    bool operator==(const Location &other) const {
        return file == other.file && line == other.line &&
               column == other.column;
    }

    static std::unique_ptr<Location>
    fromSourceLocation(const ASTContext &Context, SourceLocation loc);
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

    // 项目所在绝对目录。所有不在这个目录下的函数都会被排除
    std::string projectDirectory;

    bool isUnderProject(const std::string &file) {
        return file.compare(0, projectDirectory.size(), projectDirectory) == 0;
    }

    std::map<std::string, std::set<std::string>> callGraph;

    int functionCnt;
    std::map<std::string, int> idOfFunction;
    std::vector<NamedLocation> functionLocations;

    int getIdOfFunction(const std::string &signature) {
        auto it = idOfFunction.find(signature);
        if (it == idOfFunction.end()) {
            return -1;
        }
        return it->second;
    }

    ICFG icfg;
};

SourceLocation getEndOfMacroExpansion(SourceLocation loc, ASTContext &Context);

/**
 * 打印指定范围内的源码
 */
void printSourceWithinRange(ASTContext &Context, SourceRange range);

/**
 * 判断两个 json 在给定的 `fields` 中，是否完全相同。
 *
 * 注意，即使两个 json 都没有某个 field `f`，也会返回 false。
 */
bool allFieldsMatch(const json &x, const json &y,
                    const std::set<std::string> &fields);

#endif