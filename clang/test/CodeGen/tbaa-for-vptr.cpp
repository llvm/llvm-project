// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -o - -fsanitize=thread %s | FileCheck %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -o - -O1 %s | FileCheck %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -o - -O1  -relaxed-aliasing -fsanitize=thread %s | FileCheck %s
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -o - %s | FileCheck %s --check-prefix=NOTBAA
// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -o - -O2  -relaxed-aliasing %s | FileCheck %s --check-prefix=NOTBAA
//
// Check that we generate TBAA for vtable pointer loads and stores.
// When -fsanitize=thread is used TBAA should be generated at all opt levels
// even if -relaxed-aliasing is present.
struct A {
  virtual int foo() const ;
  virtual ~A();
};

void CreateA() {
  new A;
}

void CallFoo(A *a, int (A::*fp)() const) {
  a->foo();
  (a->*fp)();
}

// CHECK-LABEL: @_Z7CallFoo
// CHECK: %{{.*}} = load {{.*}} !tbaa ![[NUM:[0-9]+]]
// CHECK: br i1
// CHECK: load {{.*}}, !tbaa ![[NUM]]
//
// CHECK-LABEL: @_ZN1AC2Ev
// CHECK: store {{.*}} !tbaa ![[NUM]]
//
// CHECK: [[NUM]] = !{[[TYPE:!.*]], [[TYPE]], i64 0}
// CHECK: [[TYPE]] = !{!"vtable pointer", !{{.*}}
// NOTBAA-NOT: = !{!"Simple C/C++ TBAA"}
