#include "utils.h"

GlobalStat Global;

void requireTrue(bool condition, std::string message) {
    if (!condition) {
        llvm::errs() << "requireTrue failed: " << message << "\n";
        exit(1);
    }
}

std::string getFullSignature(const FunctionDecl *D) {
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
Location::fromSourceLocation(const ASTContext &Context,
                             const SourceLocation &loc) {
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
