#include "GenAST.h"

#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/CommonOptionsParser.h"

std::string getASTDumpFile(const std::string &file) { return file + ".ast"; }

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
