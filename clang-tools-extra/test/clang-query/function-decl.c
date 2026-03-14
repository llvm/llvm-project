// RUN: clang-query -c "match functionDecl()" %s -- | FileCheck %s

// CHECK: function-decl.c:4:1: note: "root" binds here
void foo(void) {}
