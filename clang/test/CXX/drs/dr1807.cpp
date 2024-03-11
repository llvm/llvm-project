// RUN: %clang_cc1 -std=c++98 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK,CXX98
// RUN: %clang_cc1 -std=c++11 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK,SINCE-CXX11
// RUN: %clang_cc1 -std=c++14 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK,SINCE-CXX11
// RUN: %clang_cc1 -std=c++17 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK,SINCE-CXX11
// RUN: %clang_cc1 -std=c++20 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK,SINCE-CXX11
// RUN: %clang_cc1 -std=c++23 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK,SINCE-CXX11
// RUN: %clang_cc1 -std=c++2c %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK,SINCE-CXX11

namespace dr1807 { // dr1807: 3.0
struct S {
  S() {}
  ~S() {}
};

void f() {
  S s[3];
}
}

// CHECK-LABEL:      define dso_local void @dr1807::f()
// CHECK:              invoke void @dr1807::S::S(){{.+}}
// CHECK-NEXT:         {{.+}} unwind label %lpad
// CHECK-LABEL:      lpad:
// CHECK:              br {{.+}}, label {{.+}}, label %arraydestroy.body
// CHECK-LABEL:      arraydestroy.body:         
// CHECK:              [[ARRAYDESTROY_ELEMENT:%.*]] = getelementptr {{.+}}, i64 -1
// CXX98-NEXT:         invoke void @dr1807::S::~S()({{.*}}[[ARRAYDESTROY_ELEMENT]])
// SINCE-CXX11-NEXT:   call void @dr1807::S::~S()({{.*}}[[ARRAYDESTROY_ELEMENT]])
