// RUN: not clang-repl --Xcc -Xclang --Xcc -help 2>&1 | FileCheck %s --check-prefix=HELP
// RUN: not clang-repl --Xcc -Xclang --Xcc -version 2>&1 | FileCheck %s --check-prefix=VERSION

// HELP: Help displayed
// VERSION: Version displayed