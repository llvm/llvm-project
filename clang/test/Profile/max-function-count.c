// Test that maximum function counts are set correctly.

// RUN: llvm-profdata merge %S/Inputs/max-function-count.proftext -o %t.profdata
// RUN: %clang %s -o - -mllvm -disable-llvm-optzns -emit-llvm -S -fprofile-instr-use=%t.profdata | FileCheck %s
//
int begin(int i) {
  if (i)
    return 0;
  return 1;
}

int end(int i) {
  if (i)
    return 0;
  return 1;
}

int main(int argc, const char *argv[]) {
  begin(0);
  end(1);
  end(1);
  return 0;
}
// CHECK: !{{[0-9]+}} = !{i32 1, !"MaxFunctionCount", i32 2}
