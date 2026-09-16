// RUN: %clang_cc1 -std=c++23 -disable-llvm-passes -emit-llvm %s -triple x86_64-pc-windows-msvc -fms-extensions -o - | FileCheck -check-prefix=CHECK-MSVC %s
// RUN: %clang_cc1 -std=c++23 -disable-llvm-passes -emit-llvm %s -triple x86_64-unknown-linux-gnu -fms-extensions -o - | FileCheck -check-prefix=CHECK-LINUX %s

// CHECK-MSVC: define {{.*}} @"?foo@@YAXXZ"() #[[ATTR_FOO:[0-9]+]]
// CHECK-LINUX: define {{.*}} @_Z3foov() #[[ATTR_FOO:[0-9]+]]
// Add another attribute to prevent overlapping sets.
[[msvc::forceinline, gnu::hot]] void foo() {}

void bar();

void call_lambda() {
  auto lambda = [] [[msvc::forceinline]] () { bar(); };
  lambda();
}

// CHECK-MSVC: define internal void @"??R<lambda_0>@?0??call_lambda@@YAXXZ@QEBA?A?<auto>@@XZ"{{.*}} #[[ATTR_LAMBDA:[0-9]+]]
// CHECK-LINUX: define internal void @"_ZZ11call_lambdavENK3$_0clEv"{{.*}} #[[ATTR_LAMBDA:[0-9]+]]

void call_bar() {
// CHECK-MSVC-LABEL: define {{.*}} @"?call_bar@@YAXXZ"()
// CHECK-MSVC: call void @"?bar@@YAXXZ"() #[[ATTR_CALLSITE:[0-9]+]]
// CHECK-LINUX-LABEL: define {{.*}} @_Z8call_barv()
// CHECK-LINUX: call void @_Z3barv() #[[ATTR_CALLSITE:[0-9]+]]
  [[msvc::forceinline_calls]] bar();
}

// CHECK-MSVC-DAG: attributes #[[ATTR_FOO]] = { alwaysinline hot {{.*}}}
// CHECK-MSVC-DAG: attributes #[[ATTR_LAMBDA]] = { alwaysinline {{.*}}}
// CHECK-MSVC-DAG: attributes #[[ATTR_CALLSITE]] = { alwaysinline }

// CHECK-LINUX-DAG: attributes #[[ATTR_FOO]] = { alwaysinline hot {{.*}}}
// CHECK-LINUX-DAG: attributes #[[ATTR_LAMBDA]] = { alwaysinline {{.*}}}
// CHECK-LINUX-DAG: attributes #[[ATTR_CALLSITE]] = { alwaysinline }
