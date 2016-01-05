// Test that function entry counts are set correctly.

// RUN: llvm-profdata merge %S/Inputs/func-entry.proftext -o %t.profdata
// RUN: %clang %s -o - -mllvm -disable-llvm-optzns -emit-llvm -S -fprofile-instr-use=%t.profdata | FileCheck %s

void foo(void);

// CHECK: @foo() #{{[0-9]}} !prof [[FOO:![0-9]+]]
void foo() { return; }

// CHECK: @main() #{{[0-9]}} !prof [[MAIN:![0-9]+]]
int main() {
  int i;
  for (i = 0; i < 10000; i++) foo();
  return 0;
}

// CHECK: [[FOO]] = !{!"function_entry_count", i64 1000}
// CHECK: [[MAIN]] = !{!"function_entry_count", i64 1}
