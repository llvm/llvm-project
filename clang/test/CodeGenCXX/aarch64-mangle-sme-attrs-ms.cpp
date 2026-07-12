// RUN: %clang_cc1 -triple aarch64-windows-msvc -target-feature +sve -target-feature +sme -target-feature +sme2 %s -emit-llvm -o - | FileCheck %s

//
// Streaming-Mode Attributes
//

// CHECK: define dso_local void @"?fn_streaming@@YAXP6A__clang_sme_attr1XXZ@Z"
void fn_streaming(void (*foo)() __arm_streaming) { foo(); }

// CHECK: define dso_local void @"?fn_streaming_compatible@@YAXP6A__clang_sme_attr2HXZ@Z"
void fn_streaming_compatible(int (*foo)() __arm_streaming_compatible) { foo(); }

//
// __arm_agnostic("sme_za_state") Attribute
//

// CHECK: define dso_local void @"?fn_sme_za_state_agnostic@@YAXP6A__clang_sme_attr256XXZ@Z"
void fn_sme_za_state_agnostic(void (*foo)() __arm_agnostic("sme_za_state")) { foo(); }

// CHECK: define dso_local void @"?fn_sme_za_state_streaming_agnostic@@YAXP6A__clang_sme_attr257XXZ@Z"
void fn_sme_za_state_streaming_agnostic(void (*foo)() __arm_streaming __arm_agnostic("sme_za_state")) { foo(); }

//
// No SME Attributes
//

// CHECK: define dso_local void @"?no_sme_attrs@@YAXP6AXXZ@Z"
void no_sme_attrs(void (*foo)()) { foo(); }

// CHECK: define dso_local void @"?locally_streaming_caller@@YAXP6AXXZ@Z"
__arm_locally_streaming void locally_streaming_caller(void (*foo)()) { foo(); }

// CHECK: define dso_local void @"?streaming_caller@@YA__clang_sme_attr1XXZ"
void streaming_caller() __arm_streaming {}

// CHECK: define dso_local void @"?za_shared_caller@@YA__clang_sme_attr8XXZ"
void za_shared_caller() __arm_in("za") {}

// CHECK: define dso_local void @"?zt0_shared_caller@@YA__clang_sme_attr96XXZ"
void zt0_shared_caller() __arm_out("zt0") {}
