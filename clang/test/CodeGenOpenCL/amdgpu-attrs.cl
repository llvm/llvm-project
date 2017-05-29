// RUN: %clang_cc1 -triple amdgcn-- -target-cpu tahiti -O0 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O0 -emit-llvm -verify -o - %s | FileCheck -check-prefix=X86 %s

__attribute__((amdgpu_flat_work_group_size(0, 0))) // expected-no-diagnostics
kernel void flat_work_group_size_0_0() {}
__attribute__((amdgpu_waves_per_eu(0))) // expected-no-diagnostics
kernel void waves_per_eu_0() {}
__attribute__((amdgpu_waves_per_eu(0, 0))) // expected-no-diagnostics
kernel void waves_per_eu_0_0() {}
__attribute__((amdgpu_num_sgpr(0))) // expected-no-diagnostics
kernel void num_sgpr0() {}
__attribute__((amdgpu_num_vgpr(0))) // expected-no-diagnostics
kernel void num_vgpr0() {}

__attribute__((amdgpu_flat_work_group_size(0, 0), amdgpu_waves_per_eu(0))) // expected-no-diagnostics
kernel void flat_work_group_size_0_0_waves_per_eu_0() {}
__attribute__((amdgpu_flat_work_group_size(0, 0), amdgpu_waves_per_eu(0, 0))) // expected-no-diagnostics
kernel void flat_work_group_size_0_0_waves_per_eu_0_0() {}
__attribute__((amdgpu_flat_work_group_size(0, 0), amdgpu_num_sgpr(0))) // expected-no-diagnostics
kernel void flat_work_group_size_0_0_num_sgpr_0() {}
__attribute__((amdgpu_flat_work_group_size(0, 0), amdgpu_num_vgpr(0))) // expected-no-diagnostics
kernel void flat_work_group_size_0_0_num_vgpr_0() {}
__attribute__((amdgpu_waves_per_eu(0), amdgpu_num_sgpr(0))) // expected-no-diagnostics
kernel void waves_per_eu_0_num_sgpr_0() {}
__attribute__((amdgpu_waves_per_eu(0), amdgpu_num_vgpr(0))) // expected-no-diagnostics
kernel void waves_per_eu_0_num_vgpr_0() {}
__attribute__((amdgpu_waves_per_eu(0, 0), amdgpu_num_sgpr(0))) // expected-no-diagnostics
kernel void waves_per_eu_0_0_num_sgpr_0() {}
__attribute__((amdgpu_waves_per_eu(0, 0), amdgpu_num_vgpr(0))) // expected-no-diagnostics
kernel void waves_per_eu_0_0_num_vgpr_0() {}
__attribute__((amdgpu_num_sgpr(0), amdgpu_num_vgpr(0))) // expected-no-diagnostics
kernel void num_sgpr_0_num_vgpr_0() {}

__attribute__((amdgpu_flat_work_group_size(0, 0), amdgpu_waves_per_eu(0), amdgpu_num_sgpr(0))) // expected-no-diagnostics
kernel void flat_work_group_size_0_0_waves_per_eu_0_num_sgpr_0() {}
__attribute__((amdgpu_flat_work_group_size(0, 0), amdgpu_waves_per_eu(0), amdgpu_num_vgpr(0))) // expected-no-diagnostics
kernel void flat_work_group_size_0_0_waves_per_eu_0_num_vgpr_0() {}
__attribute__((amdgpu_flat_work_group_size(0, 0), amdgpu_waves_per_eu(0, 0), amdgpu_num_sgpr(0))) // expected-no-diagnostics
kernel void flat_work_group_size_0_0_waves_per_eu_0_0_num_sgpr_0() {}
__attribute__((amdgpu_flat_work_group_size(0, 0), amdgpu_waves_per_eu(0, 0), amdgpu_num_vgpr(0))) // expected-no-diagnostics
kernel void flat_work_group_size_0_0_waves_per_eu_0_0_num_vgpr_0() {}

__attribute__((amdgpu_flat_work_group_size(0, 0), amdgpu_waves_per_eu(0), amdgpu_num_sgpr(0), amdgpu_num_vgpr(0))) // expected-no-diagnostics
kernel void flat_work_group_size_0_0_waves_per_eu_0_num_sgpr_0_num_vgpr_0() {}
__attribute__((amdgpu_flat_work_group_size(0, 0), amdgpu_waves_per_eu(0, 0), amdgpu_num_sgpr(0), amdgpu_num_vgpr(0))) // expected-no-diagnostics
kernel void flat_work_group_size_0_0_waves_per_eu_0_0_num_sgpr_0_num_vgpr_0() {}

__attribute__((amdgpu_flat_work_group_size(32, 64))) // expected-no-diagnostics
kernel void flat_work_group_size_32_64() {
// CHECK: define amdgpu_kernel void @flat_work_group_size_32_64() [[FLAT_WORK_GROUP_SIZE_32_64:#[0-9]+]]
}
__attribute__((amdgpu_waves_per_eu(2))) // expected-no-diagnostics
kernel void waves_per_eu_2() {
// CHECK: define amdgpu_kernel void @waves_per_eu_2() [[WAVES_PER_EU_2:#[0-9]+]]
}
__attribute__((amdgpu_waves_per_eu(2, 4))) // expected-no-diagnostics
kernel void waves_per_eu_2_4() {
// CHECK: define amdgpu_kernel void @waves_per_eu_2_4() [[WAVES_PER_EU_2_4:#[0-9]+]]
}
__attribute__((amdgpu_num_sgpr(32))) // expected-no-diagnostics
kernel void num_sgpr_32() {
// CHECK: define amdgpu_kernel void @num_sgpr_32() [[NUM_SGPR_32:#[0-9]+]]
}
__attribute__((amdgpu_num_vgpr(64))) // expected-no-diagnostics
kernel void num_vgpr_64() {
// CHECK: define amdgpu_kernel void @num_vgpr_64() [[NUM_VGPR_64:#[0-9]+]]
}

__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2))) // expected-no-diagnostics
kernel void flat_work_group_size_32_64_waves_per_eu_2() {
// CHECK: define amdgpu_kernel void @flat_work_group_size_32_64_waves_per_eu_2() [[FLAT_WORK_GROUP_SIZE_32_64_WAVES_PER_EU_2:#[0-9]+]]
}
__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2, 4))) // expected-no-diagnostics
kernel void flat_work_group_size_32_64_waves_per_eu_2_4() {
// CHECK: define amdgpu_kernel void @flat_work_group_size_32_64_waves_per_eu_2_4() [[FLAT_WORK_GROUP_SIZE_32_64_WAVES_PER_EU_2_4:#[0-9]+]]
}
__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_num_sgpr(32))) // expected-no-diagnostics
kernel void flat_work_group_size_32_64_num_sgpr_32() {
// CHECK: define amdgpu_kernel void @flat_work_group_size_32_64_num_sgpr_32() [[FLAT_WORK_GROUP_SIZE_32_64_NUM_SGPR_32:#[0-9]+]]
}
__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_num_vgpr(64))) // expected-no-diagnostics
kernel void flat_work_group_size_32_64_num_vgpr_64() {
// CHECK: define amdgpu_kernel void @flat_work_group_size_32_64_num_vgpr_64() [[FLAT_WORK_GROUP_SIZE_32_64_NUM_VGPR_64:#[0-9]+]]
}
__attribute__((amdgpu_waves_per_eu(2), amdgpu_num_sgpr(32))) // expected-no-diagnostics
kernel void waves_per_eu_2_num_sgpr_32() {
// CHECK: define amdgpu_kernel void @waves_per_eu_2_num_sgpr_32() [[WAVES_PER_EU_2_NUM_SGPR_32:#[0-9]+]]
}
__attribute__((amdgpu_waves_per_eu(2), amdgpu_num_vgpr(64))) // expected-no-diagnostics
kernel void waves_per_eu_2_num_vgpr_64() {
// CHECK: define amdgpu_kernel void @waves_per_eu_2_num_vgpr_64() [[WAVES_PER_EU_2_NUM_VGPR_64:#[0-9]+]]
}
__attribute__((amdgpu_waves_per_eu(2, 4), amdgpu_num_sgpr(32))) // expected-no-diagnostics
kernel void waves_per_eu_2_4_num_sgpr_32() {
// CHECK: define amdgpu_kernel void @waves_per_eu_2_4_num_sgpr_32() [[WAVES_PER_EU_2_4_NUM_SGPR_32:#[0-9]+]]
}
__attribute__((amdgpu_waves_per_eu(2, 4), amdgpu_num_vgpr(64))) // expected-no-diagnostics
kernel void waves_per_eu_2_4_num_vgpr_64() {
// CHECK: define amdgpu_kernel void @waves_per_eu_2_4_num_vgpr_64() [[WAVES_PER_EU_2_4_NUM_VGPR_64:#[0-9]+]]
}
__attribute__((amdgpu_num_sgpr(32), amdgpu_num_vgpr(64))) // expected-no-diagnostics
kernel void num_sgpr_32_num_vgpr_64() {
// CHECK: define amdgpu_kernel void @num_sgpr_32_num_vgpr_64() [[NUM_SGPR_32_NUM_VGPR_64:#[0-9]+]]
}

__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2), amdgpu_num_sgpr(32)))
kernel void flat_work_group_size_32_64_waves_per_eu_2_num_sgpr_32() {
// CHECK: define amdgpu_kernel void @flat_work_group_size_32_64_waves_per_eu_2_num_sgpr_32() [[FLAT_WORK_GROUP_SIZE_32_64_WAVES_PER_EU_2_NUM_SGPR_32:#[0-9]+]]
}
__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2), amdgpu_num_vgpr(64)))
kernel void flat_work_group_size_32_64_waves_per_eu_2_num_vgpr_64() {
// CHECK: define amdgpu_kernel void @flat_work_group_size_32_64_waves_per_eu_2_num_vgpr_64() [[FLAT_WORK_GROUP_SIZE_32_64_WAVES_PER_EU_2_NUM_VGPR_64:#[0-9]+]]
}
__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2, 4), amdgpu_num_sgpr(32)))
kernel void flat_work_group_size_32_64_waves_per_eu_2_4_num_sgpr_32() {
// CHECK: define amdgpu_kernel void @flat_work_group_size_32_64_waves_per_eu_2_4_num_sgpr_32() [[FLAT_WORK_GROUP_SIZE_32_64_WAVES_PER_EU_2_4_NUM_SGPR_32:#[0-9]+]]
}
__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2, 4), amdgpu_num_vgpr(64)))
kernel void flat_work_group_size_32_64_waves_per_eu_2_4_num_vgpr_64() {
// CHECK: define amdgpu_kernel void @flat_work_group_size_32_64_waves_per_eu_2_4_num_vgpr_64() [[FLAT_WORK_GROUP_SIZE_32_64_WAVES_PER_EU_2_4_NUM_VGPR_64:#[0-9]+]]
}

__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2), amdgpu_num_sgpr(32), amdgpu_num_vgpr(64))) // expected-no-diagnostics
kernel void flat_work_group_size_32_64_waves_per_eu_2_num_sgpr_32_num_vgpr_64() {
// CHECK: define amdgpu_kernel void @flat_work_group_size_32_64_waves_per_eu_2_num_sgpr_32_num_vgpr_64() [[FLAT_WORK_GROUP_SIZE_32_64_WAVES_PER_EU_2_NUM_SGPR_32_NUM_VGPR_64:#[0-9]+]]
}
__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2, 4), amdgpu_num_sgpr(32), amdgpu_num_vgpr(64))) // expected-no-diagnostics
kernel void flat_work_group_size_32_64_waves_per_eu_2_4_num_sgpr_32_num_vgpr_64() {
// CHECK: define amdgpu_kernel void @flat_work_group_size_32_64_waves_per_eu_2_4_num_sgpr_32_num_vgpr_64() [[FLAT_WORK_GROUP_SIZE_32_64_WAVES_PER_EU_2_4_NUM_SGPR_32_NUM_VGPR_64:#[0-9]+]]
}

__attribute__((reqd_work_group_size(32, 2, 1))) // expected-no-diagnostics
kernel void reqd_work_group_size_32_2_1() {
// CHECK: define amdgpu_kernel void @reqd_work_group_size_32_2_1() [[FLAT_WORK_GROUP_SIZE_64_64:#[0-9]+]]
}
__attribute__((reqd_work_group_size(32, 2, 1), amdgpu_flat_work_group_size(16, 128))) // expected-no-diagnostics
kernel void reqd_work_group_size_32_2_1_flat_work_group_size_16_128() {
// CHECK: define amdgpu_kernel void @reqd_work_group_size_32_2_1_flat_work_group_size_16_128() [[FLAT_WORK_GROUP_SIZE_16_128:#[0-9]+]]
}


// Make sure this is silently accepted on other targets.
// X86-NOT: "amdgpu-flat-work-group-size"
// X86-NOT: "amdgpu-waves-per-eu"
// X86-NOT: "amdgpu-num-vgpr"
// X86-NOT: "amdgpu-num-sgpr"

// CHECK-NOT: "amdgpu-flat-work-group-size"="0,0"
// CHECK-NOT: "amdgpu-waves-per-eu"="0"
// CHECK-NOT: "amdgpu-waves-per-eu"="0,0"
// CHECK-NOT: "amdgpu-num-sgpr"="0"
// CHECK-NOT: "amdgpu-num-vgpr"="0"

// CHECK-DAG: attributes [[FLAT_WORK_GROUP_SIZE_32_64]] = { noinline nounwind optnone "amdgpu-flat-work-group-size"="32,64"
// CHECK-DAG: attributes [[FLAT_WORK_GROUP_SIZE_64_64]] = { noinline nounwind optnone "amdgpu-flat-work-group-size"="64,64"
// CHECK-DAG: attributes [[FLAT_WORK_GROUP_SIZE_16_128]] = { noinline nounwind optnone "amdgpu-flat-work-group-size"="16,128"
// CHECK-DAG: attributes [[WAVES_PER_EU_2]] = { noinline nounwind optnone "amdgpu-waves-per-eu"="2"
// CHECK-DAG: attributes [[WAVES_PER_EU_2_4]] = { noinline nounwind optnone "amdgpu-waves-per-eu"="2,4"
// CHECK-DAG: attributes [[NUM_SGPR_32]] = { noinline nounwind optnone "amdgpu-num-sgpr"="32"
// CHECK-DAG: attributes [[NUM_VGPR_64]] = { noinline nounwind optnone "amdgpu-num-vgpr"="64"

// CHECK-DAG: attributes [[FLAT_WORK_GROUP_SIZE_32_64_WAVES_PER_EU_2]] = { noinline nounwind optnone "amdgpu-flat-work-group-size"="32,64" "amdgpu-waves-per-eu"="2"
// CHECK-DAG: attributes [[FLAT_WORK_GROUP_SIZE_32_64_WAVES_PER_EU_2_4]] = { noinline nounwind optnone "amdgpu-flat-work-group-size"="32,64" "amdgpu-waves-per-eu"="2,4"
// CHECK-DAG: attributes [[FLAT_WORK_GROUP_SIZE_32_64_NUM_SGPR_32]] = { noinline nounwind optnone "amdgpu-flat-work-group-size"="32,64" "amdgpu-num-sgpr"="32"
// CHECK-DAG: attributes [[FLAT_WORK_GROUP_SIZE_32_64_NUM_VGPR_64]] = { noinline nounwind optnone "amdgpu-flat-work-group-size"="32,64" "amdgpu-num-vgpr"="64"
// CHECK-DAG: attributes [[WAVES_PER_EU_2_NUM_SGPR_32]] = { noinline nounwind optnone "amdgpu-num-sgpr"="32" "amdgpu-waves-per-eu"="2"
// CHECK-DAG: attributes [[WAVES_PER_EU_2_NUM_VGPR_64]] = { noinline nounwind optnone "amdgpu-num-vgpr"="64" "amdgpu-waves-per-eu"="2"
// CHECK-DAG: attributes [[WAVES_PER_EU_2_4_NUM_SGPR_32]] = { noinline nounwind optnone "amdgpu-num-sgpr"="32" "amdgpu-waves-per-eu"="2,4"
// CHECK-DAG: attributes [[WAVES_PER_EU_2_4_NUM_VGPR_64]] = { noinline nounwind optnone "amdgpu-num-vgpr"="64" "amdgpu-waves-per-eu"="2,4"
// CHECK-DAG: attributes [[NUM_SGPR_32_NUM_VGPR_64]] = { noinline nounwind optnone "amdgpu-num-sgpr"="32" "amdgpu-num-vgpr"="64"

// CHECK-DAG: attributes [[FLAT_WORK_GROUP_SIZE_32_64_WAVES_PER_EU_2_NUM_SGPR_32]] = { noinline nounwind optnone "amdgpu-flat-work-group-size"="32,64" "amdgpu-num-sgpr"="32" "amdgpu-waves-per-eu"="2"
// CHECK-DAG: attributes [[FLAT_WORK_GROUP_SIZE_32_64_WAVES_PER_EU_2_NUM_VGPR_64]] = { noinline nounwind optnone "amdgpu-flat-work-group-size"="32,64" "amdgpu-num-vgpr"="64" "amdgpu-waves-per-eu"="2"
// CHECK-DAG: attributes [[FLAT_WORK_GROUP_SIZE_32_64_WAVES_PER_EU_2_4_NUM_SGPR_32]] = { noinline nounwind optnone "amdgpu-flat-work-group-size"="32,64" "amdgpu-num-sgpr"="32" "amdgpu-waves-per-eu"="2,4"
// CHECK-DAG: attributes [[FLAT_WORK_GROUP_SIZE_32_64_WAVES_PER_EU_2_4_NUM_VGPR_64]] = { noinline nounwind optnone "amdgpu-flat-work-group-size"="32,64" "amdgpu-num-vgpr"="64" "amdgpu-waves-per-eu"="2,4"

// CHECK-DAG: attributes [[FLAT_WORK_GROUP_SIZE_32_64_WAVES_PER_EU_2_NUM_SGPR_32_NUM_VGPR_64]] = { noinline nounwind optnone "amdgpu-flat-work-group-size"="32,64" "amdgpu-num-sgpr"="32" "amdgpu-num-vgpr"="64" "amdgpu-waves-per-eu"="2"
// CHECK-DAG: attributes [[FLAT_WORK_GROUP_SIZE_32_64_WAVES_PER_EU_2_4_NUM_SGPR_32_NUM_VGPR_64]] = { noinline nounwind optnone "amdgpu-flat-work-group-size"="32,64" "amdgpu-num-sgpr"="32" "amdgpu-num-vgpr"="64" "amdgpu-waves-per-eu"="2,4"
