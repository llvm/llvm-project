// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: echo 'int  b;' > %t.dir/test.cpp

// Header lines from diff and svn may carry a tab-separated timestamp.
// RUN: cd %t.dir && not %clang_format_diff -p1 -style=LLVM \
// RUN:   < %S/Inputs/clang-format-diff-timestamp.diff | FileCheck %s
// CHECK: -int  b;
// CHECK-NEXT: +int b;
