// RUN: %clang_cc1 -E %s | FileCheck %s

#line 21 ""
int foo() { return 42; }

#line 4 "bug.c" 
int bar() { return 21; }

// CHECK: # 21 ""
// CHECK: int foo() { return 42; }
// CHECK: # 4 "bug.c"
// CHECK: int bar() { return 21; }
