// RUN: %clang_cc1 -triple nvptx-unknown-unknown -O3 -S -o - %s -emit-llvm | FileCheck %s
// RUN: %clang_cc1 -triple nvptx64-unknown-unknown -O3 -S -o - %s -emit-llvm | FileCheck %s

int bar(int a) {
  int result;
  // CHECK: call i32 asm sideeffect "{ {{.*}}
  asm __volatile__ ("{ \n\t"
                    ".reg .pred \t%%p1; \n\t"
                    ".reg .pred \t%%p2; \n\t"
                    "setp.ne.u32 \t%%p1, %1, 0; \n\t"
                    "vote.any.pred \t%%p2, %%p1; \n\t"
                    "selp.s32 \t%0, 1, 0, %%p2; \n\t"
                    "}" : "=r"(result) : "r"(a));
  return result;
}
