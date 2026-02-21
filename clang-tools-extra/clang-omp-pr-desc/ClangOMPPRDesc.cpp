#include <cstdlib>
#include <iostream>
#include <sstream>

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: clang-omp-pr-desc <PR_NUMBER> [--spec-path <path>]\n";
        return 1;
    }

    // Reconstruct command string
    std::ostringstream command;
    command << "omp-pr-summary describe";

    for (int i = 1; i < argc; ++i) {
        command << " \"" << argv[i] << "\"";  // quote to preserve spaces in paths
    }

    int result = std::system(command.str().c_str());
    return result;
}
