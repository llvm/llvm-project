#include <regex>

#include "GenAST.h"

#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/CommonOptionsParser.h"

std::string getASTDumpFile(const std::string &file) {
    return file + ".path-gen-ast";
}

/**
 * 判断参数是否形如 -Dxxx"xxx，即中部包含没有闭合的引号
 */
bool needsJsonEscapingFixing(const std::string &arg) {
    const static std::string middle = "[-=A-Za-z0-9_]";
    const static std::regex pattern(
        fmt::format("^-D{}+\"{}+$", middle, middle));
    return std::regex_match(arg, pattern);
}

/**
 * 对于参数 -DPAGER_ENV="LESS=FRX LV=-c"，LLVM会把它拆分成两个参数。
 * 该函数修复这样的参数，把它们重新组合成原始状态。
 */
ArgumentsAdjuster getFixJsonEscapingAdjuster() {
    return [](const CommandLineArguments &Args, StringRef Filename) {
        CommandLineArguments result;
        for (int i = 0; i < Args.size(); i++) {
            auto &a = Args[i];
            if (needsJsonEscapingFixing(a) && i + 1 < Args.size()) {
                auto &b = Args[i + 1];
                logger.warn("Fixing wrongly escaped entry, combining:");
                logger.warn("  {}", a);
                logger.warn("  {}", b);
                result.push_back(a + " " + b);
                i++;
                continue;
            }
            result.push_back(a);
        }
        return result;
    };
}

/**
 * 删除 Clang 不认识的参数。
 * 对于使用 GCC 编译的项目，可能包含 Clang 不认识的参数。
 */
ArgumentsAdjuster getRmoveUnknownArgumentsAdjuster() {
    const std::set<std::string> unknownArgPrefixes = {
        "-mindirect-branch-register",
        "-mpreferred-stack-boundary=",
        "-mrecord-mcount",
        "-mindirect-branch=thunk-extern",
        "-mindirect-branch=thunk-inline",

        "-fno-allow-store-data-races",
        "-fconserve-stack",
        "-falign-jumps=",
        "-fno-code-hoisting",
        "-fsched-pressure",
    };

    auto unknownArgAdj = [unknownArgPrefixes](const CommandLineArguments &Args,
                                              StringRef Filename) {
        CommandLineArguments result;
        for (const auto &arg : Args) {
            bool needIgnore = false;
            for (const auto &prefix : unknownArgPrefixes) {
                if (arg.find(prefix) == 0) {
                    needIgnore = true;
                    break;
                }
            }
            if (!needIgnore)
                result.push_back(arg);
        }
        return result;
    };

    // also suppress all warnings
    // 一部分来自不认识的 Warning flag，用 -Wno-unknown-warning-option 消除
    auto suppressWarningAdj =
        getInsertArgumentAdjuster("-w", ArgumentInsertPosition::BEGIN);

    return combineAdjusters(unknownArgAdj, suppressWarningAdj);
}

ArgumentsAdjuster getEmitAstAdjuster() {
    ArgumentsAdjuster identityAdjuster = [](const CommandLineArguments &Args,
                                            StringRef Filename) {
        return Args;
    };

    ArgumentsAdjuster addAstOptionToFrontAdjuster =
        [](const CommandLineArguments &Args, StringRef Filename) {
            std::string output = getASTDumpFile(Filename.str());
            return getInsertArgumentAdjuster(
                {"-emit-ast", "-o", output.c_str()},
                ArgumentInsertPosition::BEGIN)(Args, Filename);
        };

    ArgumentsAdjuster useClangAdjuster = [](const CommandLineArguments &Args,
                                            StringRef Filename) {
        CommandLineArguments result(Args);
        const std::string originalCompiler = result[0];
        bool useCpp = false;
        if (originalCompiler == "cc") {
            useCpp = false;
        } else if (originalCompiler == "c++") {
            useCpp = true;
        } else {
            requireTrue(false, "Unknown compiler: " + originalCompiler);
        }
        result[0] = useCpp ? "clang++" : "clang";
        return result;
    };

    std::vector<ArgumentsAdjuster> adjusters = {
        getClangStripOutputAdjuster(), // 删除 -o 开头，用于指定输出文件的参数
        getFixJsonEscapingAdjuster(),
        getRmoveUnknownArgumentsAdjuster(),
        addAstOptionToFrontAdjuster, // 添加 -emit-ast -o FILE.ast 这样的参数
        useClangAdjuster,            // 指定编译器为 Clang
    };

    ArgumentsAdjuster result = identityAdjuster;
    for (auto adjuster : adjusters) {
        result = combineAdjusters(result, adjuster);
    }
    return result;
}

std::unique_ptr<CompilationDatabase>
getCompilationDatabaseWithASTEmit(fs::path buildPath) {
    auto cb = std::make_unique<ArgumentsAdjustingCompilations>(
        getCompilationDatabase(buildPath));

    logger.info("Adjusting compilation commands to emit AST ...");
    cb->appendArgumentsAdjuster(getEmitAstAdjuster());

    return cb;
}
