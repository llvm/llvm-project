// Check for an error while -new-name argument has not been passed to
// clang-rename.
// RUN: not clang-refactor-test rename-initiate -at=%s:1:11 %s 2>&1 | FileCheck %s
// CHECK: clang-refactor-test: for the -new-name option: must be specified at least once
