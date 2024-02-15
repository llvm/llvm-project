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
