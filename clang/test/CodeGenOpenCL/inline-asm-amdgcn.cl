// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -emit-llvm -O0 -o - -triple amdgcn %s | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -O0 -o - -triple spirv64-amd-amdhsa %s | FileCheck %s

typedef float float32 __attribute__((ext_vector_type(32)));

kernel void test_long(int arg0) {
  long v15_16;
  // CHECK: call{{.*}} i64 asm sideeffect "v_lshlrev_b64 v[15:16], 0, $0", "={v[15:16]},v"
  __asm volatile("v_lshlrev_b64 v[15:16], 0, %0" : "={v[15:16]}"(v15_16) : "v"(arg0));
}

kernel void test_agpr() {
  float32 acc_c;
  float reg_a;
  float reg_b;
  float32 reg_c;
  // CHECK:  call{{.*}} <32 x float> asm "v_mfma_f32_32x32x1f32 $0, $1, $2, $3", "=a,v,v,a,~{a0},~{a1},~{a2},~{a3},~{a4},~{a5},~{a6},~{a7},~{a8},~{a9},~{a10},~{a11},~{a12},~{a13},~{a14},~{a15},~{a16},~{a17},~{a18},~{a19},~{a20},~{a21},~{a22},~{a23},~{a24},~{a25},~{a26},~{a27},~{a28},~{a29},~{a30},~{a31}"
  __asm ("v_mfma_f32_32x32x1f32 %0, %1, %2, %3"
         : "=a"(acc_c)
         : "v"(reg_a), "v"(reg_b), "a"(reg_c)
         : "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
           "a8", "a9", "a10", "a11", "a12", "a13", "a14", "a15",
           "a16", "a17", "a18", "a19", "a20", "a21", "a22", "a23",
           "a24", "a25", "a26", "a27", "a28", "a29", "a30", "a31");

   // CHECK: call{{.*}} <32 x float> asm sideeffect "v_mfma_f32_32x32x1f32 a[0:31], $0, $1, a[0:31]", "={a[0:31]},v,v,{a[0:31]}"
  __asm volatile("v_mfma_f32_32x32x1f32 a[0:31], %0, %1, a[0:31]"
                 : "={a[0:31]}"(acc_c)
                 : "v"(reg_a),"v"(reg_b), "{a[0:31]}"(reg_c));

  // CHECK: call{{.*}} float asm "v_accvgpr_read_b32 $0, $1", "={a1},{a1}"
  __asm ("v_accvgpr_read_b32 %0, %1"
         : "={a1}"(reg_a)
         : "{a1}"(reg_b));
}

kernel void test_constraint_DA() {
  const long x = 0x200000001;
  int res;
  // CHECK: call{{.*}} i32 asm sideeffect "v_mov_b32 $0, $1 & 0xFFFFFFFF", "=v,^DA"(i64 8589934593)
  __asm volatile("v_mov_b32 %0, %1 & 0xFFFFFFFF" : "=v"(res) : "DA"(x));
}

kernel void test_constraint_DB() {
  const long x = 0x200000001;
  int res;
  // CHECK: call{{.*}} i32 asm sideeffect "v_mov_b32 $0, $1 & 0xFFFFFFFF", "=v,^DB"(i64 8589934593)
  __asm volatile("v_mov_b32 %0, %1 & 0xFFFFFFFF" : "=v"(res) : "DB"(x));
}
