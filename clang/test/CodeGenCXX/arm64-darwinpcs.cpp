// RUN: %clang_cc1 -triple arm64-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-linux-gnu -emit-llvm -o - %s -target-abi darwinpcs | FileCheck %s --check-prefix=CHECK-DARWIN

void test_extensions(bool a, char b, short c) {}
// CHECK: define{{.*}} void @_Z15test_extensionsbcs(i1 %a, i8 %b, i16 %c)
// CHECK-DARWIN: define{{.*}} void @_Z15test_extensionsbcs(i1 zeroext %a, i8 signext %b, i16 signext %c)

struct Empty {};
void test_empty(Empty e) {}
// CHECK: define{{.*}} void @_Z10test_empty5Empty(i8
// CHECK-DARWIN: define{{.*}} void @_Z10test_empty5Empty()

struct HFA {
  float a[3];
};
