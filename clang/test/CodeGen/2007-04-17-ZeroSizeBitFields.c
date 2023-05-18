// PR 1332
// RUN: %clang_cc1 %s -emit-llvm -o /dev/null

struct Z { int a:1; int :0; int c:1; } z;
