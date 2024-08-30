// RUN: %clang -S -O2 -emit-llvm -fextend-lifetimes %s -o - | FileCheck %s
// REQUIRES: asserts
//
// We are checking that the fake.use calls for i, j and k appear
// in a particular order. It is not the order itself that is important
// but that it remains the same between different test runs.

// CHECK:       call {{.*}}void (...) @llvm.fake.use(i32 %k)
// CHECK-NEXT:  call {{.*}}void (...) @llvm.fake.use(i32 %j)
// CHECK-NEXT:  call {{.*}}void (...) @llvm.fake.use(i32 %i)

extern void bar();
void foo(int i, int j, int k)
{
   for (int l = 0; l < i; l++) {
      bar();
   }
}
