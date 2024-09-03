// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -emit-llvm -o - %s | FileCheck %s


// CHECK-LABEL: define {{[^@]+}}@n_callee._Msve
// CHECK-SAME: () #[[ATTR0:[0-9]+]] {
//
// CHECK-LABEL: define {{[^@]+}}@n_callee._Msimd
// CHECK-SAME: () #[[ATTR1:[0-9]+]] {
//
__arm_locally_streaming __attribute__((target_clones("sve", "simd"))) void n_callee(void) {}
// CHECK-LABEL: define {{[^@]+}}@n_callee._Msme2
// CHECK-SAME: () #[[ATTR2:[0-9]+]] {
//
__attribute__((target_version("sme2"))) void n_callee(void) {}
// CHECK-LABEL: define {{[^@]+}}@n_callee.default
// CHECK-SAME: () #[[ATTR3:[0-9]+]] {
//
__attribute__((target_version("default"))) void n_callee(void) {}


// CHECK-LABEL: define {{[^@]+}}@s_callee._Msve
// CHECK-SAME: () #[[ATTR4:[0-9]+]] {
//
// CHECK-LABEL: define {{[^@]+}}@s_callee._Msimd
// CHECK-SAME: () #[[ATTR5:[0-9]+]] {
//
__attribute__((target_clones("sve", "simd"))) void s_callee(void) __arm_streaming {}
// CHECK-LABEL: define {{[^@]+}}@s_callee._Msme2
// CHECK-SAME: () #[[ATTR6:[0-9]+]] {
//
__arm_locally_streaming __attribute__((target_version("sme2"))) void s_callee(void) __arm_streaming {}
// CHECK-LABEL: define {{[^@]+}}@s_callee.default
// CHECK-SAME: () #[[ATTR7:[0-9]+]] {
//
__attribute__((target_version("default"))) void s_callee(void) __arm_streaming {}


// CHECK-LABEL: define {{[^@]+}}@sc_callee._Msve
// CHECK-SAME: () #[[ATTR8:[0-9]+]] {
//
// CHECK-LABEL: define {{[^@]+}}@sc_callee._Msimd
// CHECK-SAME: () #[[ATTR9:[0-9]+]] {
//
__attribute__((target_clones("sve", "simd"))) void sc_callee(void) __arm_streaming_compatible {}
// CHECK-LABEL: define {{[^@]+}}@sc_callee._Msme2
// CHECK-SAME: () #[[ATTR10:[0-9]+]] {
//
__arm_locally_streaming __attribute__((target_version("sme2"))) void sc_callee(void) __arm_streaming_compatible {}
// CHECK-LABEL: define {{[^@]+}}@sc_callee.default
// CHECK-SAME: () #[[ATTR11:[0-9]+]] {
//
__attribute__((target_version("default"))) void sc_callee(void) __arm_streaming_compatible {}


// CHECK-LABEL: define {{[^@]+}}@n_caller
// CHECK-SAME: () #[[ATTR3:[0-9]+]] {
// CHECK:    call void @n_callee()
// CHECK:    call void @s_callee() #[[ATTR12:[0-9]+]]
// CHECK:    call void @sc_callee() #[[ATTR13:[0-9]+]]
//
void n_caller(void) {
  n_callee();
  s_callee();
  sc_callee();
}


// CHECK-LABEL: define {{[^@]+}}@s_caller
// CHECK-SAME: () #[[ATTR7:[0-9]+]] {
// CHECK:    call void @n_callee()
// CHECK:    call void @s_callee() #[[ATTR12]]
// CHECK:    call void @sc_callee() #[[ATTR13]]
//
void s_caller(void) __arm_streaming {
  n_callee();
  s_callee();
  sc_callee();
}


// CHECK-LABEL: define {{[^@]+}}@sc_caller
// CHECK-SAME: () #[[ATTR11:[0-9]+]] {
// CHECK:    call void @n_callee()
// CHECK:    call void @s_callee() #[[ATTR12]]
// CHECK:    call void @sc_callee() #[[ATTR13]]
//
void sc_caller(void) __arm_streaming_compatible {
  n_callee();
  s_callee();
  sc_callee();
}


// CHECK: attributes #[[ATTR0:[0-9]+]] = {{.*}} "aarch64_pstate_sm_body"
// CHECK: attributes #[[ATTR1:[0-9]+]] = {{.*}} "aarch64_pstate_sm_body"
// CHECK: attributes #[[ATTR2:[0-9]+]] = {{.*}}
// CHECK: attributes #[[ATTR3]] = {{.*}}
// CHECK: attributes #[[ATTR4:[0-9]+]] = {{.*}} "aarch64_pstate_sm_enabled"
// CHECK: attributes #[[ATTR5:[0-9]+]] = {{.*}} "aarch64_pstate_sm_enabled"
// CHECK: attributes #[[ATTR6:[0-9]+]] = {{.*}} "aarch64_pstate_sm_body" "aarch64_pstate_sm_enabled"
// CHECK: attributes #[[ATTR7]] = {{.*}} "aarch64_pstate_sm_enabled"
// CHECK: attributes #[[ATTR8:[0-9]+]] = {{.*}} "aarch64_pstate_sm_compatible"
// CHECK: attributes #[[ATTR9:[0-9]+]] = {{.*}} "aarch64_pstate_sm_compatible"
// CHECK: attributes #[[ATTR10]] = {{.*}} "aarch64_pstate_sm_body" "aarch64_pstate_sm_compatible"
// CHECK: attributes #[[ATTR11]] = {{.*}} "aarch64_pstate_sm_compatible"
// CHECK: attributes #[[ATTR12]] = {{.*}} "aarch64_pstate_sm_enabled"
// CHECK: attributes #[[ATTR13]] = {{.*}} "aarch64_pstate_sm_compatible"
