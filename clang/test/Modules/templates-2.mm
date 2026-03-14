// RUN: rm -rf %t
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++11 -x objective-c++ -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs -verify %s -Wno-objc-root-class
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++11 -x objective-c++ -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs -emit-llvm %s -o - -Wno-objc-root-class | FileCheck %s
// expected-no-diagnostics
// REQUIRES: x86-registered-target
@import templates_top;

struct TestEmitDefaultedSpecialMembers {
  EmitDefaultedSpecialMembers::SmallVector<char, 256> V;
};

@import templates_left;

void testEmitDefaultedSpecialMembers() {
  EmitDefaultedSpecialMembers::SmallString<256> V;
  // CHECK: call {{.*}} @_ZN27EmitDefaultedSpecialMembers11SmallStringILj256EEC1Ev(
  // CHECK: call {{.*}} @_ZN27EmitDefaultedSpecialMembers11SmallStringILj256EED1Ev(
}

// CHECK-LABEL: define {{.*}} @_ZN27EmitDefaultedSpecialMembers11SmallStringILj256EEC1Ev(
// CHECK: call {{.*}} @_ZN27EmitDefaultedSpecialMembers11SmallStringILj256EEC2Ev(

// CHECK-LABEL: define {{.*}} @_ZN27EmitDefaultedSpecialMembers11SmallStringILj256EED1Ev(
// CHECK: call {{.*}} @_ZN27EmitDefaultedSpecialMembers11SmallStringILj256EED2Ev(

// CHECK-LABEL: define linkonce_odr void @_ZN27EmitDefaultedSpecialMembers11SmallStringILj256EEC2Ev(
// CHECK: call void @_ZN27EmitDefaultedSpecialMembers11SmallVectorIcLj256EEC2Ev(
// CHECK-LABEL: define linkonce_odr void @_ZN27EmitDefaultedSpecialMembers11SmallVectorIcLj256EEC2Ev(
// CHECK: call void @_ZN27EmitDefaultedSpecialMembers15SmallVectorImplIcEC2Ev(
// CHECK-LABEL: define linkonce_odr void @_ZN27EmitDefaultedSpecialMembers15SmallVectorImplIcEC2Ev(

// CHECK-LABEL: define linkonce_odr void @_ZN27EmitDefaultedSpecialMembers11SmallStringILj256EED2Ev(
// CHECK: call void @_ZN27EmitDefaultedSpecialMembers11SmallVectorIcLj256EED2Ev(
// CHECK-LABEL: define linkonce_odr void @_ZN27EmitDefaultedSpecialMembers11SmallVectorIcLj256EED2Ev(
// CHECK: call void @_ZN27EmitDefaultedSpecialMembers15SmallVectorImplIcED2Ev(
// CHECK-LABEL: define linkonce_odr void @_ZN27EmitDefaultedSpecialMembers15SmallVectorImplIcED2Ev(
