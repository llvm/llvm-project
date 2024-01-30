#include "utils.h"

fs::path BUILD_PATH;

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