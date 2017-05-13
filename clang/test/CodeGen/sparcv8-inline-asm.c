// RUN: %clang_cc1 -triple sparc-unknown-unknown -emit-llvm %s -o - | FileCheck %s

// CHECK: define float @fabsf(float %a)
// CHECK: %{{.*}} = call float asm sideeffect "fabss $1, $0;", "=e,f"(float %{{.*}}) #1
float fabsf(float a) {
  float res;
  __asm __volatile__("fabss  %1, %0;"
                     : /* reg out*/ "=e"(res)
                     : /* reg in */ "f"(a));
  return res;
}
