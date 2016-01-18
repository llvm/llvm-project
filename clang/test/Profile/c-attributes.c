// Test that instrumentation based profiling sets function attributes correctly.

// RUN: llvm-profdata merge %S/Inputs/c-attributes.proftext -o %t.profdata
// RUN: %clang %s -o - -mllvm -disable-llvm-optzns -emit-llvm -S -fprofile-instr-use=%t.profdata | FileCheck %s

extern int atoi(const char *);

// CHECK: hot_100_percent(i32{{.*}}%i) [[HOT:#[0-9]+]]
void hot_100_percent(int i) {
  while (i > 0)
    i--;
}

// CHECK: hot_40_percent(i32{{.*}}%i) [[HOT]]
void hot_40_percent(int i) {
  while (i > 0)
    i--;
}

// CHECK: normal_func(i32{{.*}}%i) [[NORMAL:#[0-9]+]]
void normal_func(int i) {
  while (i > 0)
    i--;
}

// CHECK: cold_func(i32{{.*}}%i) [[COLD:#[0-9]+]]
void cold_func(int i) {
  while (i > 0)
    i--;
}

// CHECK: attributes [[HOT]] = { inlinehint nounwind {{.*}} }
// CHECK: attributes [[NORMAL]] = { nounwind {{.*}} }
// CHECK: attributes [[COLD]] = { cold nounwind {{.*}} }

int main(int argc, const char *argv[]) {
  int max = atoi(argv[1]);
  int i;
  for (i = 0; i < max; i++)
    hot_100_percent(i);
  for (i = 0; i < max * 4 / 10; i++)
    hot_40_percent(i);
  for (i = 0; i < max * 2 / 10; i++)
    normal_func(i);
  for (i = 0; i < max / 200; i++)
    cold_func(i);
  return 0;
}
