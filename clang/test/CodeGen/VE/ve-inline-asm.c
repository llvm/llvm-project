// REQUIRES: ve-registered-target
// RUN: %clang_cc1 -triple ve-linux-gnu -emit-llvm -o - %s | FileCheck %s

void r(long v) {
  long b;
  asm("lea %0, 256(%1)"
      : "=r"(b)
      : "r"(v));
  // CHECK: %1 = call i64 asm "lea $0, 256($1)", "=r,r"(i64 %0)
}

void v(char *ptr, char *ptr2) {
  typedef double __vr __attribute__((__vector_size__(2048)));
  __vr a;
  asm("vld %0, 8, %1"
      : "=v"(a)
      : "r"(ptr));
  asm("vst %0, 8, %1"
      :
      : "v"(a), "r"(ptr2));
  // CHECK: %1 = call <256 x double> asm "vld $0, 8, $1", "=v,r"(i8* %0)
  // CHECK: call void asm sideeffect "vst $0, 8, $1", "v,r"(<256 x double> %2, i8* %3)
}
