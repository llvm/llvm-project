// RUN: %clang_cc1 -fvisibility=hidden "-triple" "x86_64-apple-darwin8.0.0" -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-10_4 %s
// RUN: %clang_cc1 -fvisibility=hidden "-triple" "x86_64-apple-darwin9.0.0" -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-10_5 %s
// RUN: %clang_cc1 -fvisibility=hidden "-triple" "x86_64-apple-darwin10.0.0" -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-10_6 %s

// CHECK-10_4-LABEL: define hidden void @f2
// CHECK-10_5-LABEL: define hidden void @f2
// CHECK-10_6-LABEL: define hidden void @f2
void f2(void);
void f2(void) { }

// CHECK-10_4-LABEL: define hidden void @f3
// CHECK-10_5-LABEL: define hidden void @f3
// CHECK-10_6-LABEL: define hidden void @f3
void f3(void) __attribute__((availability(macosx,introduced=10.5)));
void f3(void) { }

// CHECK-10_4: declare extern_weak void @f0
// CHECK-10_5: declare void @f0
// CHECK-10_6: declare void @f0
void f0(void) __attribute__((availability(macosx,introduced=10.5)));

// CHECK-10_4: declare extern_weak void @f1
// CHECK-10_5: declare extern_weak void @f1
// CHECK-10_6: declare void @f1
void f1(void) __attribute__((availability(macosx,introduced=10.6)));

void test(void) {
  f0();
  f1();
  f2();
}
