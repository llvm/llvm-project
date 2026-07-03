// RUN: %clang_cc1 -triple aarch64 -target-feature +sme -target-feature +sme2 -x c++ -std=c++20  -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK

struct TestStruct;

__arm_new("za", "zt0") void test(TestStruct& TS,
  void (TestStruct::*streaming_member_ptr)() __arm_streaming,
  void (TestStruct::*streaming_compat_member)() __arm_streaming_compatible,
  void (TestStruct::*arm_in_member)() __arm_in("za", "zt0"),
  void (TestStruct::*arm_inout_member)() __arm_inout("za", "zt0"),
  void (TestStruct::*arm_preserves_member)() __arm_preserves("za", "zt0"),
  void (TestStruct::*arm_agnostic_member)() __arm_agnostic("sme_za_state")) {

  // CHECK: call void %{{.*}} [[STREAMING_MEMBER_CALL_ATTRS:#.+]]
  (TS.*streaming_member_ptr)();

  // CHECK: call void %{{.*}} [[STREAMING_COMPAT_MEMBER_CALL_ATTRS:#.+]]
  (TS.*streaming_compat_member)();

  // CHECK: call void %{{.*}} [[ARM_IN_MEMBER_CALL_ATTRS:#.+]]
  (TS.*arm_in_member)();

  // CHECK: call void %{{.*}} [[ARM_INOUT_MEMBER_CALL_ATTRS:#.+]]
  (TS.*arm_inout_member)();

  // CHECK: call void %{{.*}} [[ARM_PRESERVES_MEMBER_CALL_ATTRS:#.+]]
  (TS.*arm_preserves_member)();

  // CHECK: call void %{{.*}} [[ARM_AGNOSTIC_MEMBER_CALL_ATTRS:#.+]]
  (TS.*arm_agnostic_member)();
}

// CHECK: attributes [[STREAMING_MEMBER_CALL_ATTRS]] = { "aarch64_pstate_sm_enabled" }
// CHECK: attributes [[STREAMING_COMPAT_MEMBER_CALL_ATTRS]] = { "aarch64_pstate_sm_compatible" }
// CHECK: attributes [[ARM_IN_MEMBER_CALL_ATTRS]] = { "aarch64_in_za" "aarch64_in_zt0" }
// CHECK: attributes [[ARM_INOUT_MEMBER_CALL_ATTRS]] = { "aarch64_inout_za" "aarch64_inout_zt0" }
// CHECK: attributes [[ARM_PRESERVES_MEMBER_CALL_ATTRS]] = { "aarch64_preserves_za" "aarch64_preserves_zt0" }
// CHECK: attributes [[ARM_AGNOSTIC_MEMBER_CALL_ATTRS]] = { "aarch64_za_state_agnostic" }
