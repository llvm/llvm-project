// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: cd %t.dir
//
// RUN: %clang_profgen=P_RAW -o %t -O3 %s
// RUN: %run %t P_RAW
// RUN: llvm-profdata merge -o %t.profdata P_RAW
// RUN: %clang_profuse=%t.profdata -o - -S -emit-llvm %s | FileCheck %s --check-prefix=FE
//
// RUN: %clang_pgogen=I_RAW -o %t.2 %s
// RUN: %run %t.2 I_RAW
// RUN: llvm-profdata merge -o %t2.profdata I_RAW
// RUN: %clang_profuse=%t2.profdata -o - -S -emit-llvm %s | FileCheck %s --check-prefix=IR

void bar() {}
int main(int argc, const char *argv[]) {
  // FE: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !prof ![[PD1:[0-9]+]]
  // IR: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !prof ![[PD1:[0-9]+]]
  if (argc < 2)
    return 1;
  bar();
  return 0;
}
// FE: ![[PD1]] = !{!"branch_weights", i32 1, i32 2}
// IR: ![[PD1]] = !{!"branch_weights", i32 0, i32 1}
