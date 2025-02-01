// RUN: %clang_cc1 -loader-replaceable-function=override_me -emit-llvm -std=c11 -o - %s | FileCheck %s

// CHECK: define dso_local void @override_me() #0
void override_me() {}

// CHECK: define dso_local void @dont_override_me() #1
void dont_override_me() {}

// CHECK: attributes #0 = { 
// CHECK-SAME: loader-replaceable

// CHECK: attributes #1 = { 
// CHECK-NOT: loader-replaceable
// CHECK-SAME: }
