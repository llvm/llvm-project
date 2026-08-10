// RUN: %clang_cc1 -E %s | grep xxx-xxx

#define foo(return) return-return

foo(xxx)

