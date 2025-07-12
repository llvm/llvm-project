// RUN: clang-query -c "match \
// RUN:   functionDecl( \
// RUN:       hasName( \
// RUN:           \"foo\", \
// RUN:       ), \
// RUN:   ) \
// RUN: " %s -- | FileCheck %s

// CHECK: trailing-comma.c:10:1: note: "root" binds here
void foo(void) {}
