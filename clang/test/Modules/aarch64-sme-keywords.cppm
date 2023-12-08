// REQUIRES: aarch64-registered-target
//
// RUN: rm -rf %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -triple aarch64 -target-feature +sme %t/A.cppm -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -triple aarch64 -target-feature +sme -fprebuilt-module-path=%t -I%t %t/Use.cpp -emit-llvm
// RUN: cat %t/Use.ll | FileCheck %s

//--- A.cppm
module;
export module A;

export void f_streaming(void) __arm_streaming { }
export void f_streaming_compatible(void) __arm_streaming_compatible { }
export void f_shared_za(void) __arm_shared_za { }
export void f_preserves_za(void) __arm_preserves_za { }

//--- Use.cpp
// expected-no-diagnostics
import A;

// CHECK: define dso_local void @_Z18f_shared_za_callerv() #[[SHARED_ZA_DEF:[0-9]+]] {
// CHECK: entry:
// CHECK:   call void @_ZW1A11f_shared_zav() #[[SHARED_ZA_USE:[0-9]+]]
// CHECK:   call void @_ZW1A14f_preserves_zav() #[[PRESERVES_ZA_USE:[0-9]+]]
// CHECK:   ret void
// CHECK: }
//
// CHECK:declare void @_ZW1A11f_shared_zav() #[[SHARED_ZA_DECL:[0-9]+]]
//
// CHECK:declare void @_ZW1A14f_preserves_zav() #[[PRESERVES_ZA_DECL:[0-9]+]]
//
// CHECK:; Function Attrs: mustprogress noinline nounwind optnone
// CHECK:define dso_local void @_Z21f_nonstreaming_callerv() #[[NORMAL_DEF:[0-9]+]] {
// CHECK:entry:
// CHECK:  call void @_ZW1A11f_streamingv() #[[STREAMING_USE:[0-9]+]]
// CHECK:  call void @_ZW1A22f_streaming_compatiblev() #[[STREAMING_COMPATIBLE_USE:[0-9]+]]
// CHECK:  ret void
// CHECK:}
//
// CHECK:declare void @_ZW1A11f_streamingv() #[[STREAMING_DECL:[0-9]+]]
//
// CHECK:declare void @_ZW1A22f_streaming_compatiblev() #[[STREAMING_COMPATIBLE_DECL:[0-9]+]]
//
// CHECK-DAG: attributes #[[SHARED_ZA_DEF]] = {{{.*}} "aarch64_pstate_za_shared" {{.*}}}
// CHECK-DAG: attributes #[[SHARED_ZA_DECL]] = {{{.*}} "aarch64_pstate_za_shared" {{.*}}}
// CHECK-DAG: attributes #[[PRESERVES_ZA_DECL]] = {{{.*}} "aarch64_pstate_za_preserved" {{.*}}}
// CHECK-DAG: attributes #[[NORMAL_DEF]] = {{{.*}}}
// CHECK-DAG: attributes #[[STREAMING_DECL]] = {{{.*}} "aarch64_pstate_sm_enabled" {{.*}}}
// CHECK-DAG: attributes #[[STREAMING_COMPATIBLE_DECL]] = {{{.*}} "aarch64_pstate_sm_compatible" {{.*}}}
// CHECK-DAG: attributes #[[SHARED_ZA_USE]] = { "aarch64_pstate_za_shared" }
// CHECK-DAG: attributes #[[PRESERVES_ZA_USE]] = { "aarch64_pstate_za_preserved" }
// CHECK-DAG: attributes #[[STREAMING_USE]] = { "aarch64_pstate_sm_enabled" }
// CHECK-DAG: attributes #[[STREAMING_COMPATIBLE_USE]] = { "aarch64_pstate_sm_compatible" }

void f_shared_za_caller(void) __arm_shared_za {
  f_shared_za();
  f_preserves_za();
}

void f_nonstreaming_caller(void) {
  f_streaming();
  f_streaming_compatible();
}
