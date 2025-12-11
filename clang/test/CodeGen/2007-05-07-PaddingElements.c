// PR 1278
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix %s -emit-llvm -o - | FileCheck %s --check-prefix=AIX

// CHECK: %struct.s = type { double, i32 }
// AIX: %struct.s = type { double, i32, [4 x i8] }
struct s {
  double d1;
  int s1;
};

struct s foo(void) {
  struct s S = {1.1, 2};
  return S;
}
