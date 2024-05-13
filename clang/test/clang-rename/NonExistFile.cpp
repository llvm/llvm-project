// RUN: not clang-rename -offset=0 -new-name=bar non-existing-file 2>&1 | FileCheck %s
// CHECK: clang-rename: non-existing-file does not exist.
