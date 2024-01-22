// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme2 \
// RUN:   -S -disable-O0-optnone -Werror -emit-llvm -o - %s \
// RUN: | opt -S -passes=mem2reg \
// RUN: | opt -S -passes=inline \
// RUN: | FileCheck %s

// Test the attributes for ZT0 and their mappings to LLVM IR.

extern "C" {

// CHECK-LABEL: @in_zt0()
// CHECK-SAME: #[[ZT0_IN:[0-9]+]]
void in_zt0() __arm_in("zt0") { }

// CHECK-LABEL: @out_zt0()
// CHECK-SAME: #[[ZT0_OUT:[0-9]+]]
void out_zt0() __arm_out("zt0") { }

// CHECK-LABEL: @inout_zt0()
// CHECK-SAME: #[[ZT0_INOUT:[0-9]+]]
void inout_zt0() __arm_inout("zt0") { }

// CHECK-LABEL: @preserves_zt0()
// CHECK-SAME: #[[ZT0_PRESERVED:[0-9]+]]
void preserves_zt0() __arm_preserves("zt0") { }

// CHECK-LABEL: @new_zt0()
// CHECK-SAME: #[[ZT0_NEW:[0-9]+]]
__arm_new("zt0") void new_zt0() { }

// CHECK-LABEL: @in_za_zt0()
// CHECK-SAME: #[[ZA_ZT0_IN:[0-9]+]]
void in_za_zt0() __arm_in("za", "zt0") { }

// CHECK-LABEL: @out_za_zt0()
// CHECK-SAME: #[[ZA_ZT0_OUT:[0-9]+]]
void out_za_zt0() __arm_out("za", "zt0") { }

// CHECK-LABEL: @inout_za_zt0()
// CHECK-SAME: #[[ZA_ZT0_INOUT:[0-9]+]]
void inout_za_zt0() __arm_inout("za", "zt0") { }

// CHECK-LABEL: @preserves_za_zt0()
// CHECK-SAME: #[[ZA_ZT0_PRESERVED:[0-9]+]]
void preserves_za_zt0() __arm_preserves("za", "zt0") { }

// CHECK-LABEL: @new_za_zt0()
// CHECK-SAME: #[[ZA_ZT0_NEW:[0-9]+]]
__arm_new("za", "zt0") void new_za_zt0() { }

}

// CHECK: attributes #[[ZT0_IN]] = {{{.*}} "aarch64_in_zt0" {{.*}}}
// CHECK: attributes #[[ZT0_OUT]] = {{{.*}} "aarch64_out_zt0" {{.*}}}
// CHECK: attributes #[[ZT0_INOUT]] = {{{.*}} "aarch64_inout_zt0" {{.*}}}
// CHECK: attributes #[[ZT0_PRESERVED]] = {{{.*}} "aarch64_preserves_zt0" {{.*}}}
// CHECK: attributes #[[ZT0_NEW]] = {{{.*}} "aarch64_new_zt0" {{.*}}}
