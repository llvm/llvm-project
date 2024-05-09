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
#include <random>
#include <set>
#include <stack>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#include "ICFG.h"

#include "lib/indicators.hpp"
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
typedef std::map<std::string, std::set<std::unique_ptr<FunctionInfo>>> fif;

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

template <> struct fmt::formatter<json> : fmt::formatter<std::string> {
    auto format(const json &j, format_context &ctx) const
        -> decltype(ctx.out()) {
        return formatter<std::string>::format(
            j.dump(4, ' ', false, json::error_handler_t::replace), ctx);
    }
};

template <> struct fmt::formatter<ordered_json> : fmt::formatter<std::string> {
    auto format(const ordered_json &j, format_context &ctx) const
        -> decltype(ctx.out()) {
        return formatter<std::string>::format(
            j.dump(4, ' ', false, json::error_handler_t::replace), ctx);
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
    std::set<std::string> allFiles; // cc.json 中的所有文件，不包括头文件

    // 项目所在绝对目录。所有不在这个目录下的函数都会被排除
    std::string projectDirectory;

    bool isUnderProject(const std::string &file) {
        return file.compare(0, projectDirectory.size(), projectDirectory) == 0;
    }

    std::map<std::string, std::set<std::string>> callGraph;

    int functionCnt;
    std::map<std::string, std::map<std::string, int>> fileAndIdOfFunction;
    std::vector<NamedLocation> functionLocations;

    /**
     * 获取函数的 fid
     *
     * 目前的mapping是：signature -> {file -> fid}
     * 其中 file 是函数定义所在的文件名，这是为了防止同名函数的冲突，
     * 例如不同文件中的 main() 函数。
     *
     * 如果不提供函数的定义位置，就会选择第一个找到的函数。
     */
    int getIdOfFunction(const std::string &signature,
                        const std::string &file = "") {
        auto _it = fileAndIdOfFunction.find(signature);
        if (_it == fileAndIdOfFunction.end()) {
            return -1;
        }
        // 获取 signature 对应的 file -> fid 的 mapping
        std::map<std::string, int> &fidOfFile = _it->second;
        requireTrue(fidOfFile.size() > 0);
        if (file.empty()) {
            // 默认用第一个找到的函数
            auto it = fidOfFile.begin();
            if (fidOfFile.size() > 1) {
                const auto &sourceFile = it->first;
                auto fid = it->second;
                logger.warn("Function {} is defined in multiple files and no "
                            "specific one is chosen!",
                            signature);
                logger.warn("  Using the one in {} with fid {}", sourceFile,
                            fid);
            }
            return it->second;
        } else {
            auto it = fidOfFile.find(file);
            if (it == fidOfFile.end()) {
                logger.error("No record of function {} in file {}!", signature,
                             file);
                return -1;
            }
            return it->second;
        }
    }

    ICFG icfg;

    std::string clangPath, clangppPath;

    std::set<ordered_json> npeSuspectedSources; // 每个元素是可疑的 source
    // 对于加入 npeSuspectedSources 的 p = foo()，
    // 把 foo() map 到集合中的迭代器上，用于之后的删除
    std::map<std::string, std::vector<std::set<ordered_json>::iterator>>
        npeSuspectedSourcesItMap;
    // 判断函数是否包含 return NULL 语句
    std::map<int, bool> functionReturnsNull; // fid -> whether func returns null
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

/**
 * 判断路径是否对应存在的目录
 *
 * 来自 https://stackoverflow.com/q/18100097/11938767
 */
bool dirExists(const std::string &path);

/**
 * 判断文件是否存在（并且只是普通文件，不是链接、目录等）
 */
bool fileExists(const std::string &path);

/**
 * 在指定目录运行程序，并返回程序的返回值
 */
int run_program(const std::vector<std::string> &args, const std::string &pwd);

/**
 * 设置用于生成 AST 的 clang & clang++ 编译器路径
 */
void setClangPath(const char *argv0);

/**
 * 生成一个在 [a, b] 范围内的均匀分布的随机数
 */
int randomInt(int a, int b);

template <typename T>
std::optional<typename std::set<T>::iterator>
returnOnInsertSuccess(std::set<T> &s, const T &e) {
    auto p = s.insert(e);
    if (p.second)
        return p.first;
    return std::nullopt;
}

/**
 * 水池采样。判断是否要将元素 element 加入集合 reservoir，采样大小为
 * sampleSize。
 */
template <typename T>
std::optional<typename std::set<T>::iterator>
reservoirSamplingAddElement(std::set<T> &reservoir, const T &element,
                            int sampleSize) {
    // 当前集合大小
    int currentSize = reservoir.size();

    // 如果当前集合大小小于样本大小，直接将元素添加到集合中
    if (currentSize < sampleSize) {
        return returnOnInsertSuccess(reservoir, element);
    } else {
        // 否则，以概率 sampleSize / currentSize 将元素替换掉集合中的一个元素
        int replaceIndex = randomInt(0, currentSize - 1);
        if (replaceIndex >= sampleSize) // 不替换
            return std::nullopt;

        // 元素存在，当作替换成功
        if (reservoir.find(element) != reservoir.end())
            return reservoir.find(element);

        // 随机选中一个元素，并将其替换为新元素
        auto it = reservoir.begin();
        std::advance(it, replaceIndex);
        reservoir.erase(it);
        return returnOnInsertSuccess(reservoir, element);
    }
    return std::nullopt;
}

class ProgressBar {
  private:
    indicators::BlockProgressBar bar;
    const int totalSize;

  public:
    ProgressBar(std::string msg, int totalSize, int barWidth = 60)
        : totalSize(totalSize), //
          bar{
              indicators::option::BarWidth{barWidth},
              indicators::option::Start{"["},
              indicators::option::End{"]"},
              indicators::option::ShowElapsedTime{true},
              indicators::option::ShowRemainingTime{true},
              indicators::option::MaxProgress{totalSize},
              indicators::option::PrefixText{msg + " "},
          } {}

    void tick() {
        bar.tick();
        int current = bar.current();
        std::string postfix = fmt::format("{}/{}", current, totalSize);
        bar.set_option(indicators::option::PostfixText{postfix});
    }

    void done() { bar.mark_as_completed(); }
};

#endif