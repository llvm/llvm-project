#include "CompilationDatabase.h"

std::unique_ptr<CompilationDatabase>
getCompilationDatabase(fs::path buildPath) {
    logger.info("Reading compilation database from: {}", buildPath);
    std::string errorMsg;
    std::unique_ptr<CompilationDatabase> cb =
        CompilationDatabase::autoDetectFromDirectory(buildPath.string(),
                                                     errorMsg);
    // FIXME: 在 npe/input.json 下会报错
    // #include "clang/Tooling/JSONCompilationDatabase.h"
    // JSONCompilationDatabase::loadFromFile(
    //     buildPath.string(), errorMsg, JSONCommandLineSyntax::AutoDetect);
    if (!cb) {
        logger.error("Error while trying to load compilation database: {}",
                     errorMsg);
        exit(1);
    }
    logger.info("There are {} entries for {} files in total",
                cb->getAllCompileCommands().size(), cb->getAllFiles().size());
    return cb;
}

void dumpCompileCommand(const CompileCommand &cmd) {
    logger.info("CompileCommand for: {}", cmd.Filename);
    logger.warn("  dir: {}", cmd.Directory);
    logger.warn("  cmd: {}", fmt::join(cmd.CommandLine, " "));
}

void dumpCompilationDatabase(const CompilationDatabase &cb) {
    auto commands = cb.getAllCompileCommands();
    for (auto cmd : commands) {
        dumpCompileCommand(cmd);
    }
}
