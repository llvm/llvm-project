#include "utils.h"

#include <sys/wait.h>
#include <unistd.h>

GlobalStat Global;

spdlog::logger &logger = *spdlog::default_logger();

void requireTrue(bool condition, std::string message) {
    if (!condition) {
        throw std::runtime_error("requireTrue failed: " + message);
    }
}

std::string getFullSignature(const FunctionDecl *D) {
    if (!D)
        return "";
    // 问题: parameter type 里没有 namespace 的信息
    std::string fullSignature = D->getQualifiedNameAsString();
    fullSignature += "(";
    bool first = true;
    for (auto &p : D->parameters()) {
        if (first)
            first = false;
        else
            fullSignature += ", ";
        fullSignature += p->getType().getAsString();
    }
    fullSignature += ")";
    return fullSignature;
}

void dumpSourceLocation(const std::string &msg, const ASTContext &Context,
                        const SourceLocation &loc) {
    auto &SM = Context.getSourceManager();

    llvm::errs() << msg << ": ";
    if (loc.isInvalid()) {
        llvm::errs() << "Invalid location!\n";
        return;
    }

    if (loc.isMacroID()) {
        llvm::errs() << "MACRO S:" //
                     << SM.getSpellingLineNumber(loc) << ":"
                     << SM.getSpellingColumnNumber(loc) << " "
                     << "E:";
    }

    llvm::errs() << SM.getExpansionLineNumber(loc) << ":"
                 << SM.getExpansionColumnNumber(loc) << " ";

    // if (loc.isMacroID()) {
    //     llvm::errs() << "I:";
    //     CharSourceRange range = SM.getImmediateExpansionRange(loc);
    // }

    const FileEntry *fileEntry =
        SM.getFileEntryForID(SM.getFileID(SM.getExpansionLoc(loc)));
    if (fileEntry) {
        llvm::errs() << fileEntry->tryGetRealPathName() << " ";
    } else {
        llvm::errs() << "null ";
    }

    llvm::errs() << "\n";
}

void dumpStmt(const ASTContext &Context, const Stmt *S) {
    const auto &b = S->getBeginLoc();
    const auto &_e = S->getEndLoc();

    // 从 stmt 获取的 loc 一定 valid
    requireTrue(b.isValid() && _e.isValid());

    SourceLocation e = Lexer::getLocForEndOfToken(
        _e, 0, Context.getSourceManager(), Context.getLangOpts());

    dumpSourceLocation("> b ", Context, b);
    dumpSourceLocation("> _e", Context, _e);
    dumpSourceLocation("> e ", Context, e);

    S->dump();

    // if (b.isMacroID() && !_e.isMacroID())
    //     requireTrue(false, "b is macro but e is not");
    // if (!b.isMacroID() && _e.isMacroID())
    //     requireTrue(false, "e is macro but b is not");
}

std::unique_ptr<Location>
Location::fromSourceLocation(const ASTContext &Context, SourceLocation loc) {
    /**
     * Special handling for macro expansion location.
     *
     * Taken from SourceLocation::print() in clang/lib/Basic/SourceLocation.cpp
     */
    if (loc.isMacroID())
        loc = Context.getSourceManager().getExpansionLoc(loc);

    FullSourceLoc FullLocation = Context.getFullLoc(loc);
    if (FullLocation.isInvalid() || !FullLocation.hasManager())
        return nullptr;

    const FileEntry *fileEntry = FullLocation.getFileEntry();
    if (!fileEntry)
        return nullptr;

    StringRef _file = fileEntry->tryGetRealPathName();
    if (_file.empty())
        return nullptr;
    std::string file(_file);

    int line = FullLocation.getSpellingLineNumber();
    int column = FullLocation.getSpellingColumnNumber();

    return std::make_unique<Location>(file, line, column);
}

SourceLocation getEndOfMacroExpansion(SourceLocation loc, ASTContext &Context) {
    auto &SM = Context.getSourceManager();
    auto &LangOpts = Context.getLangOpts();
    requireTrue(loc.isValid() && loc.isMacroID());

    FileID FID = SM.getFileID(loc);
    const SrcMgr::SLocEntry *Entry = &SM.getSLocEntry(FID);
    while (Entry->getExpansion().getExpansionLocStart().isMacroID()) {
        loc = Entry->getExpansion().getExpansionLocStart();
        FID = SM.getFileID(loc);
        Entry = &SM.getSLocEntry(FID);
    }
    SourceLocation end = Entry->getExpansion().getExpansionLocEnd();
    requireTrue(end.isValid() && !end.isMacroID());
    return end;
}

void printSourceWithinRange(ASTContext &Context, SourceRange range) {
    auto &SM = Context.getSourceManager();
    auto &LO = Context.getLangOpts();

    SourceLocation b = range.getBegin();
    SourceLocation e = range.getEnd();

    if (b.isMacroID() && e.isMacroID()) {
        llvm::errs() << "!!! both beg & end are macro !!!\n";

        // b = SM.getExpansionLoc(b);
        // e = SM.getExpansionLoc(e);

        // b = SM.getSpellingLoc(b);
        // e = SM.getSpellingLoc(e);

        e = getEndOfMacroExpansion(b, Context);
        e = Lexer::getLocForEndOfToken(e, 0, SM, LO);
        requireTrue(e.isValid(), "End is invalid");

        b = SM.getExpansionLoc(b);

        auto CharRange = CharSourceRange::getCharRange(b, e);
        auto StringRep = Lexer::getSourceText(CharRange, SM, LO);
        llvm::errs() << "> " << StringRep << "\n";
    }
}

bool allFieldsMatch(const json &x, const json &y,
                    const std::set<std::string> &fields) {
    for (auto &f : fields) {
        if (!x.contains(f) || !y.contains(f) || x[f] != y[f])
            return false;
    }
    return true;
}

bool dirExists(const std::string &path) {
    struct stat info;

    if (stat(path.c_str(), &info) != 0)
        return false;
    return info.st_mode & S_IFDIR;
}

bool fileExists(const std::string &path) {
    struct stat info;

    if (stat(path.c_str(), &info) != 0)
        return false;
    return info.st_mode & S_IFREG;
}
int run_program(const std::vector<std::string> &args, const std::string &pwd) {
    // 创建 C 字符串数组来存储参数
    std::vector<char *> c_args;
    for (const auto &arg : args) {
        c_args.push_back(const_cast<char *>(arg.c_str()));
    }
    c_args.push_back(nullptr); // 添加 NULL 作为结束标志

    // 创建新的进程
    pid_t pid = fork();

    if (pid == -1) {
        // 错误处理，创建进程失败
        perror("fork");
        return -1;
    } else if (pid == 0) {
        // 在子进程中改变工作目录
        if (chdir(pwd.c_str()) == -1) {
            // 如果chdir返回-1，说明改变目录失败
            perror("chdir");
            return -1;
        }

        // 在子进程中执行程序
        if (execvp(c_args[0], c_args.data()) == -1) {
            // 如果execvp返回-1，说明执行失败
            perror("execvp");
            return -1;
        }
    } else {
        // 在父进程中等待子进程结束
        int status;
        waitpid(pid, &status, 0);

        if (WIFEXITED(status)) {
            // 子进程正常退出，返回退出状态
            return WEXITSTATUS(status);
        } else {
            // 子进程异常退出
            return -1;
        }
    }
}

void setClangPath(const char *argv0) {
    auto binPath = fs::canonical(argv0);
    logger.info("Binary path: {}", binPath);

    auto binDir = binPath.parent_path();
    logger.info("Using clang & clang++ under the same directory:");

    Global.clangPath = binDir / "clang";
    Global.clangppPath = binDir / "clang++";

    logger.info("  clang:   {}", Global.clangPath);
    requireTrue(fs::exists(Global.clangPath), "clang not found");
    logger.info("  clang++: {}", Global.clangppPath);
    requireTrue(fs::exists(Global.clangppPath), "clang not found");
}

int randomInt(int a, int b) {
    std::random_device rd;  // 用于获取随机数的设备
    std::mt19937 gen(rd()); // 使用 Mersenne Twister 算法生成随机数
    std::uniform_int_distribution<> dis(a, b);
    return dis(gen);
}

/*****************************************************************
 * 以下是没用到的函数
 *****************************************************************/

/**
 * FIXME: all files 应该只有
 * .c/.cpp，不包括头文件，所以统计会不准。以及，buildPath
 * 现在是一个 JSON 文件了，不是目录。
 */
void printCloc(const std::vector<std::string> &allFiles) {
    // save all files to "compile_files.txt" under build path
    fs::path resultFiles =
        fs::path(Global.projectDirectory) / "compile_files.txt";
    std::ofstream ofs(resultFiles);
    if (!ofs.is_open()) {
        llvm::errs() << "Error: cannot open file " << resultFiles << "\n";
        exit(1);
    }
    for (auto &file : allFiles)
        ofs << file << "\n";
    ofs.close();

    // run cloc on all files
    if (ErrorOr<std::string> P = sys::findProgramByName("cloc")) {
        std::string programPath = *P;
        std::vector<StringRef> args;
        args.push_back("cloc");
        args.push_back("--list-file");
        args.push_back(resultFiles.c_str()); // don't use .string() here
        std::string errorMsg;
        if (sys::ExecuteAndWait(programPath, args, std::nullopt, {}, 0, 0,
                                &errorMsg)) {
            llvm::errs() << "Error: " << errorMsg << "\n";
        }
    }
}