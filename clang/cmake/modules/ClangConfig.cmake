# This file allows users to call find_package(Clang) and pick up our targets.

# Clang doesn't have any CMake configuration settings yet because it mostly
# uses LLVM's. When it does, we should move this file to ClangConfig.cmake.in
# and call configure_file() on it.

# Provide all our library targets to users.
include("${CMAKE_CURRENT_LIST_DIR}/ClangTargets.cmake")
