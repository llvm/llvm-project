// RUN: %clang_cc1 -std=c++98 %s -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK,CXX98-11
// RUN: %clang_cc1 -std=c++11 %s -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK,CXX98-11
// RUN: %clang_cc1 -std=c++14 %s -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK,SINCE-CXX14
// RUN: %clang_cc1 -std=c++17 %s -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK,SINCE-CXX14
// RUN: %clang_cc1 -std=c++20 %s -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK,SINCE-CXX14
// RUN: %clang_cc1 -std=c++23 %s -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK,SINCE-CXX14
// RUN: %clang_cc1 -std=c++2c %s -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK,SINCE-CXX14

namespace dr292 { // dr292: 2.9

extern int g();

struct A {
  A(int) throw() {}
};

void f() {
  new A(g());
}

// CHECK-LABEL: define {{.*}} void @dr292::f()()
// CHECK:         %[[CALL:.+]] = call {{.*}} @operator new(unsigned long)({{.*}})
// CHECK:         invoke {{.*}} i32 @dr292::g()()
// CHECK-NEXT:           to {{.*}} unwind label %lpad
// CHECK-LABEL: lpad:
// CXX98-11:      call void @operator delete(void*)(ptr {{.*}} %[[CALL]])
// SINCE-CXX14:   call void @operator delete(void*, unsigned long)(ptr {{.*}} %[[CALL]], i64 noundef 1)
// CHECK-LABEL: eh.resume:
// CHECK-LABEL: }

} // namespace dr292
