// RUN: not clang-tidy %s -checks='-*'
// RUN: clang-tidy %s -checks='-*' --allow-no-checks | FileCheck --match-full-lines %s

// CHECK: No checks enabled.
