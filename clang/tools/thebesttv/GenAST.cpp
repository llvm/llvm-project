#include <regex>

#include "GenAST.h"

#include "clang/Frontend/CompilerInstance.h"
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
        for (size_t i = 0; i < Args.size(); i++) {
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
        "-mhard-float",

        "-fno-allow-store-data-races",
        "-fconserve-stack",
        "-falign-jumps=",
        "-fno-code-hoisting",
        "-fsched-pressure",

        // from ZRQ, project: grpc
        "--with-libxml",
        "--with-gssapi",
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
        result[0] = useCpp ? Global.clangppPath : Global.clangPath;
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

int generateASTDump(const CompileCommand &cmd) {
    int ret = run_program(cmd.CommandLine, cmd.Directory);
    if (ret != 0) {
        logger.error("Error generating AST dump for: {}", cmd.Filename);
    }
    return ret;
}

std::unique_ptr<ASTUnit> loadFromASTDump(std::string AstDumpPath) {
    if (!fileExists(AstDumpPath)) {
        logger.error("AST dump not found: {}", AstDumpPath);
        return nullptr;
    }

    auto PCHContainerOps = std::make_shared<PCHContainerOperations>();
    auto Diags = CompilerInstance::createDiagnostics(new DiagnosticOptions());
    auto HSOpts = std::make_shared<HeaderSearchOptions>();

    // TODO: 如果是 diag::err_pch_different_branch，就删除 AST，试图重新生成
    return ASTUnit::LoadFromASTFile(
        AstDumpPath, PCHContainerOps->getRawReader(), ASTUnit::LoadEverything,
        Diags, FileSystemOptions(), HSOpts);
}

std::unique_ptr<ASTUnit> createASTOfFile(std::string file) {
    std::string AstDumpPath = getASTDumpFile(file);
    if (!fileExists(AstDumpPath)) {
        // logger.warn("AST dump not found, generating from: {}", file);
        // 文件并不直接记录在 cc.json 中，很可能是头文件
        // 需要找到 ICFG 生成时，它的 AST 对应 cc.json 中的哪个文件
        if (Global.allFiles.find(file) == Global.allFiles.end()) {
            logger.warn("File has no record in cc.json (possible header): {}",
                        file);
            const auto &sourceForFile = Global.icfg.sourceForFile;
            if (sourceForFile.find(file) != sourceForFile.end()) {
                file = sourceForFile.at(file);
                AstDumpPath = getASTDumpFile(file);
                logger.warn("  Replacing with source: {}", file);
            }
        }
        auto commands = Global.cb->getCompileCommands(file);
        if (commands.empty()) {
            logger.error("No compile command found for {}!", file);
            return nullptr;
        }
        if (commands.size() > 1) {
            logger.warn(
                "Multiple compile commands found for {}, using the first one",
                file);
        }
        if (generateASTDump(commands[0]) != 0) {
            logger.error("Failed to generate AST dump for {}", file);
            return nullptr;
        }
        // logger.info("AST dump generated successfully");
    }
    return loadFromASTDump(AstDumpPath);
}

ASTFromFile::ASTFromFile(const std::string &file)
    : ASTDumpFile(getASTDumpFile(file)),
      AST(std::shared_ptr(createASTOfFile(file))) {}

std::shared_ptr<ASTFromFile> getASTOfFile(std::string file) {
    return std::make_shared<ASTFromFile>(file);
}
