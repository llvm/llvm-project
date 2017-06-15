// XFAIL: *
// TODO: Remove if unused
// Note: the run lines follow their respective tests, since line/column
// matter in this test

int variable = 0;

// RUN: not clang-refactor-test rename-initiate -at=%s:4:5 -new-name=class %s 2>&1 | FileCheck %s
// RUN: not clang-refactor-test rename-initiate -at=%s:4:5 -new-name=- %s 2>&1 | FileCheck %s
// RUN: not clang-refactor-test rename-initiate -at=%s:4:5 -new-name=var+ %s 2>&1 | FileCheck %s
// RUN: not clang-refactor-test rename-initiate -at=%s:4:5 -new-name="var " %s 2>&1 | FileCheck %s

// CHECK: error: invalid new name
