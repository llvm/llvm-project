// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

constexpr extern int cx_var = __builtin_is_constant_evaluated();

// CIR: cir.global {{.*}} @cx_var = #cir.int<1> : !s32i
// LLVM: @cx_var = {{.*}} i32 1
// OGCG: @cx_var = {{.*}} i32 1

constexpr extern float cx_var_single = __builtin_huge_valf();

// CIR: cir.global {{.*}} @cx_var_single = #cir.fp<0x7F800000> : !cir.float
// LLVM: @cx_var_single = {{.*}} float 0x7FF0000000000000
// OGCG: @cx_var_single = {{.*}} float 0x7FF0000000000000

constexpr extern long double cx_var_ld = __builtin_huge_vall();

// CIR: cir.global {{.*}} @cx_var_ld = #cir.fp<0x7FFF8000000000000000> : !cir.long_double<!cir.f80>
// LLVM: @cx_var_ld = {{.*}} x86_fp80 0xK7FFF8000000000000000
// OGCG: @cx_var_ld = {{.*}} x86_fp80 0xK7FFF8000000000000000

int is_constant_evaluated() {
  return __builtin_is_constant_evaluated();
}

// CIR: cir.func @_Z21is_constant_evaluatedv() -> !s32i
// CIR: %[[ZERO:.+]] = cir.const #cir.int<0>

// LLVM: define {{.*}}i32 @_Z21is_constant_evaluatedv()
// LLVM: %[[MEM:.+]] = alloca i32
// LLVM: store i32 0, ptr %[[MEM]]
// LLVM: %[[RETVAL:.+]] = load i32, ptr %[[MEM]]
// LLVM: ret i32 %[[RETVAL]]
// LLVM: }

// OGCG: define {{.*}}i32 @_Z21is_constant_evaluatedv()
// OGCG: ret i32 0
// OGCG: }

long double constant_fp_builtin_ld() {
  return __builtin_fabsl(-0.1L);
}

// CIR: cir.func @_Z22constant_fp_builtin_ldv() -> !cir.long_double<!cir.f80>
// CIR: %[[PONE:.+]] = cir.const #cir.fp<1.000000e-01> : !cir.long_double<!cir.f80>

// LLVM: define {{.*}}x86_fp80 @_Z22constant_fp_builtin_ldv()
// LLVM: %[[MEM:.+]] = alloca x86_fp80
// LLVM: store x86_fp80 0xK3FFBCCCCCCCCCCCCCCCD, ptr %[[MEM]]
// LLVM: %[[RETVAL:.+]] = load x86_fp80, ptr %[[MEM]]
// LLVM: ret x86_fp80 %[[RETVAL]]
// LLVM: }

// OGCG: define {{.*}}x86_fp80 @_Z22constant_fp_builtin_ldv()
// OGCG: ret x86_fp80 0xK3FFBCCCCCCCCCCCCCCCD
// OGCG: }

float constant_fp_builtin_single() {
  return __builtin_fabsf(-0.1f);
}

// CIR: cir.func @_Z26constant_fp_builtin_singlev() -> !cir.float
// CIR: %[[PONE:.+]] = cir.const #cir.fp<1.000000e-01> : !cir.float

// LLVM: define {{.*}}float @_Z26constant_fp_builtin_singlev()
// LLVM: %[[MEM:.+]] = alloca float
// LLVM: store float 0x3FB99999A0000000, ptr %[[MEM]]
// LLVM: %[[RETVAL:.+]] = load float, ptr %[[MEM]]
// LLVM: ret float %[[RETVAL]]
// LLVM: }

// OGCG: define {{.*}}float @_Z26constant_fp_builtin_singlev()
// OGCG: ret float 0x3FB99999A0000000
// OGCG: }
