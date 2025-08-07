// RUN: %clang_cc1 -triple=x86_64-pc-windows-msvc -loader-replaceable-function=override_me -loader-replaceable-function="?override_me_cpp@@YAXXZ" -emit-llvm -o - %s | FileCheck %s

// CHECK: define dso_local void @override_me() #0
extern "C" void override_me() {}

// CHECK: define dso_local void @"?override_me_cpp@@YAXXZ"() #0
void override_me_cpp() {}

// CHECK: define dso_local void @dont_override_me() #1
extern "C" void dont_override_me() {}

// CHECK: attributes #0 = { 
// CHECK-SAME: loader-replaceable

// CHECK: attributes #1 = { 
// CHECK-NOT: loader-replaceable
// CHECK-SAME: }
