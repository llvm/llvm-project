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

Location::Location() : Location("", -1, -1) {}

Location::Location(std::string file, int line, int column)
    : file(file), line(line), column(column) {}

bool Location::operator==(const Location &other) const {
    return file == other.file && line == other.line && column == other.column;
}

NamedLocation::NamedLocation() : NamedLocation("", -1, -1, "") {}

NamedLocation::NamedLocation(std::string file, int line, int column,
                             std::string name)
    : Location(file, line, column), name(name) {}

bool NamedLocation::operator==(const NamedLocation &other) const {
    return Location::operator==(other) && name == other.name;
}