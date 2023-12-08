// RUN: not %clang_cc1 %s -emit-llvm -o /dev/null

struct A X[(927 - 37) / sizeof(struct A)];
