// RUN: %clang_cpp_csi_c %s -S -o - | FileCheck %s
#include <iostream>
#include <cilk/cilk.h>

void increment(int *x, int n) {
  for (int i = 0; i < n; i++)
    x[i]++;
}

int main(int argc, char** argv) {
  int n = 1;
  if (argc == 2) n = atoi(argv[1]);

  int *x = (int*)malloc(n * sizeof(int));

  cilk_for (int i = 0; i < 1000; i++)
    increment(x, n);

  std::cout << x[0] << '\n';
}

// CHECK-LABEL: define internal{{.*}} void @__cxx_global_var_init()
// CHECK-NOT: call void @__csi_func_entry
// CHECK: ret void

// CHECK-LABEL: define internal{{.*}} void @_GLOBAL__sub_I_ctor_test.cpp()
// CHECK-NOT: call void @__csi_func_entry
// CHECK: ret void
