// RUN: %clang_cc1 %s -emit-llvm -o -

struct U { char a; short b; int c:25; char d; } u;

