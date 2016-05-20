// RUN: %clang_cc1 -triple hexagon-unknown-elf -target-feature +hvx -emit-llvm -o - %s | FileCheck %s

typedef int v64 __attribute__((__vector_size__(64)))
    __attribute__((aligned(64)));

int g;

void foo(v64 v0, v64 v1, v64 *p) {
  int r;
  v64 q0;
  asm ("%0 = vgtw(%1.w,%2.w)" : "=q"(q0) : "v"(v0), "v"(v1));
// CHECK: call <16 x i32> asm "$0 = vgtw($1.w,$2.w)", "=q,v,v"(<16 x i32>{{.*}}, <16 x i32>{{.*}})
  *p = q0;

  asm ("%0 = memw(##%1)" : "=r"(r) : "s"(&g));
// CHECK: call i32 asm "$0 = memw(##$1)", "=r,s"(i32* @g)
}
