// REQUIRES: host-supports-jit
// UNSUPPORTED: system-aix
//
// RUN: cat %s | clang-repl | FileCheck %s
// RUN: cat %s | clang-repl -Xcc -O2 | FileCheck %s

struct box { box() = default; box(int *const data) : data{data} {} int *data{}; };

box foo() { box ret; ret = new int{}; return ret; }

extern "C" int printf(const char *, ...);
printf("good");
// CHECK: good
