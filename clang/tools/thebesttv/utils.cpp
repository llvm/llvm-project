#include "utils.h"

fs::path BUILD_PATH;

std::map<std::string, std::set<const FunctionInfo *>> functionsInFile;

void requireTrue(bool condition, std::string message) {
    if (!condition) {
        llvm::errs() << "requireTrue failed: " << message << "\n";
        exit(1);
    }
}