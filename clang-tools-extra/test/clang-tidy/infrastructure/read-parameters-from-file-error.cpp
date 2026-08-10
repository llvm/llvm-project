// RUN: echo @%t.param > %t.param && not clang-tidy %s @%t.param -- 2>&1 | FileCheck %s

// CHECK: recursive expansion of
