// RUN: %clang_cc1 %s -emit-llvm -fextend-lifetimes -o %t.ll
//
// Check we don't assert when we have a return in a nested conditional and
// there is no code at the end of the function.

// CHECK: define{{.*}}main
// CHECK: call{{.*}}llvm.fake.use

void foo(int i) {
   while (0)
     if (1)
       return;
}
