// RUN: %clang_profgen=%t.bad.profraw -o %t -O3 %s
// RUN: env LLVM_PROFILE_FILE=%t.good.profraw %run %t %t.bad.profraw
// RUN: llvm-profdata merge -o %t.profdata %t.good.profraw
// RUN: %clang_profuse=%t.profdata -o - -S -emit-llvm %s | FileCheck %s

void bar () {}
int main(int argc, const char *argv[]) {
  // CHECK: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !prof ![[PD1:[0-9]+]]
  if (argc < 2)
    return 1;
  bar();
  return 0;
}
// CHECK: ![[PD1]] = !{!"branch_weights", i32 1, i32 2}
