// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sme -target-feature +sme2 %s -emit-llvm -o - | FileCheck %s

typedef __attribute__((neon_vector_type(2))) int int32x2_t;

//
// Streaming-Mode Attributes
//

// CHECK: define dso_local void @_Z12fn_streamingP11__SME_ATTRSIFvvELj1EE
void fn_streaming(void (*foo)() __arm_streaming) { foo(); }

// CHECK: define dso_local void @_Z23fn_streaming_compatibleP11__SME_ATTRSIFivELj2EE(
void fn_streaming_compatible(int (*foo)() __arm_streaming_compatible) { foo(); }

//
// ZA Attributes
//

// CHECK: define dso_local void @_Z15fn_za_preservedP11__SME_ATTRSIF11__Int32x2_tvELj32EE(
__arm_new("za") void fn_za_preserved(int32x2_t (*foo)() __arm_preserves("za")) { foo(); }

// CHECK: define dso_local void @_Z8fn_za_inP11__SME_ATTRSIFvu13__SVFloat64_tELj8EES_(
__arm_new("za") void fn_za_in(void (*foo)(__SVFloat64_t) __arm_in("za"), __SVFloat64_t x) { foo(x); }

// CHECK: define dso_local noundef i32 @_Z9fn_za_outP11__SME_ATTRSIFivELj16EE(
__arm_new("za") int fn_za_out(int (*foo)() __arm_out("za")) { return foo(); }

// CHECK: define dso_local void @_Z11fn_za_inoutP11__SME_ATTRSIFvvELj24EE(
__arm_new("za") void fn_za_inout(void (*foo)() __arm_inout("za")) { foo(); }


//
// ZT0 Attributes
//

// CHECK: define dso_local void @_Z16fn_zt0_preservedP11__SME_ATTRSIFivELj256EE(
__arm_new("zt0") void fn_zt0_preserved(int (*foo)() __arm_preserves("zt0")) { foo(); }

// CHECK: define dso_local void @_Z9fn_zt0_inP11__SME_ATTRSIFivELj64EE(
__arm_new("zt0") void fn_zt0_in(int (*foo)() __arm_in("zt0")) { foo(); }

// CHECK: define dso_local void @_Z10fn_zt0_outP11__SME_ATTRSIFivELj128EE(
__arm_new("zt0") void fn_zt0_out(int (*foo)() __arm_out("zt0")) { foo(); }

// CHECK: define dso_local void @_Z12fn_zt0_inoutP11__SME_ATTRSIFivELj192EE(
__arm_new("zt0") void fn_zt0_inout(int (*foo)() __arm_inout("zt0")) { foo(); }

//
// __arm_agnostic("sme_za_state") Attribute
//

// CHECK: define dso_local void @_Z24fn_sme_za_state_agnosticP11__SME_ATTRSIFvvELj4EE(
void fn_sme_za_state_agnostic(void (*foo)() __arm_agnostic("sme_za_state")) { foo(); }

// CHECK: define dso_local void @_Z34fn_sme_za_state_streaming_agnosticP11__SME_ATTRSIFvvELj5EE(
void fn_sme_za_state_streaming_agnostic(void (*foo)() __arm_streaming __arm_agnostic("sme_za_state")) { foo(); }

//
// Streaming-mode, ZA & ZT0 Attributes
//

// CHECK: define dso_local void @_Z17fn_all_attr_typesP11__SME_ATTRSIFivELj282EE(
__arm_new("za") __arm_new("zt0")
void fn_all_attr_types(int (*foo)() __arm_streaming_compatible __arm_inout("za") __arm_preserves("zt0"))
{ foo(); }

//
// No SME Attributes
//

// CHECK: define dso_local void @_Z12no_sme_attrsPFvvE(
void no_sme_attrs(void (*foo)()) { foo(); }

// CHECK: define dso_local void @_Z24locally_streaming_callerPFvvE(
__arm_locally_streaming void locally_streaming_caller(void (*foo)()) { foo(); }

// CHECK: define dso_local void @_Z16streaming_callerv(
void streaming_caller() __arm_streaming {}

// CHECK: define dso_local void @_Z16za_shared_callerv(
void za_shared_caller() __arm_in("za") {}

// CHECK: define dso_local void @_Z17zt0_shared_callerv(
void zt0_shared_caller() __arm_out("zt0") {}
