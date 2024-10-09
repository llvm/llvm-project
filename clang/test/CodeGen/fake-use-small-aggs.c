// RUN: %clang_cc1 %s -O2 -emit-llvm -fextend-lifetimes -disable-llvm-passes -o - | FileCheck %s
// Check that we generate a fake_use call for small aggregate types.

// CHECK-DAG:  %[[FAKEUSE1:[^ ]+]] = load{{.*}} %loc,
// CHECK-DAG:  call{{.*}}llvm.fake.use({{.*}}%[[FAKEUSE1]])
// CHECK-DAG:  %[[FAKEUSE2:[^ ]+]] = load{{.*}} %arr,
// CHECK-DAG:  call{{.*}}llvm.fake.use({{.*}}%[[FAKEUSE2]])
// CHECK-DAG:  %[[FAKEUSE3:[^ ]+]] = load{{.*}} %S,
// CHECK-DAG:  call{{.*}}llvm.fake.use({{.*}}%[[FAKEUSE3]])

struct s {
  int i;
  int j;
};

extern void inita(int *);
extern struct s inits();
void foo(struct s S)
{
   int arr[4];
   inita (arr);
   struct s loc = inits();
}

