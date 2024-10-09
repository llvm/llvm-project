// RUN: %clang %s -S -emit-llvm -fextend-lifetimes -O2 -o - -fno-discard-value-names | FileCheck %s
//
// Check we can correctly produce fake uses for function-level variables even
// when we have a return in a nested conditional and there is no code at the end
// of the function.

// CHECK-LABEL: define{{.*}}@_Z3fooi
// CHECK:         call{{.*}}llvm.fake.use(i32 %i)
// CHECK-LABEL: define{{.*}}@_ZN1C3barEi
// CHECK-DAG:     call{{.*}}llvm.fake.use(i32 %i)
// CHECK-DAG:     call{{.*}}llvm.fake.use({{.*}}%this)

void foo(int i) {
   while (0)
     if (1)
       return;
}

class C {
  void bar(int i);
};

void C::bar(int i) {
  while (0)
    if (1)
      return;
}
