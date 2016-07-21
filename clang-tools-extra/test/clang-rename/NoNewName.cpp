// Check for an error while -new-name argument has not been passed to
// clang-rename.
// RUN: not clang-rename -offset=133 %s 2>&1 | FileCheck %s
// CHECK: ERROR: no new name provided.
