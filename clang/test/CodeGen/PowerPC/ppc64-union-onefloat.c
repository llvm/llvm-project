// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefixes=ALL,GPR
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -fclang-abi-compat=23 \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefixes=ALL,UNWRAP
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -target-abi elfv2 \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefixes=ALL,UNWRAP
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefixes=ALL,UNWRAP

// ELFv1 does not unwrap unions for FPR/VR passing; ELFv2 does.

typedef float v4sf __attribute__((vector_size(16)));

typedef struct { float a; } S1;
typedef union { float a; } U1;
typedef union { double a; } UD;
typedef union { long double a; } ULD;
typedef union { v4sf v; } UV;
typedef struct { U1 u; } SU1;
typedef union { S1 s; } US1;
typedef struct { U1 a[1]; } SAU1;

// ALL-LABEL: define{{.*}} void @s1(float inreg %
void s1(S1 x) {}

// GPR-LABEL:    define{{.*}} void @u1(i32 %
// UNWRAP-LABEL: define{{.*}} void @u1(float inreg %
void u1(U1 x) {}

// GPR-LABEL:    define{{.*}} void @ud(i64 %
// UNWRAP-LABEL: define{{.*}} void @ud(double inreg %
void ud(UD x) {}

// GPR-LABEL:    define{{.*}} void @uld([1 x i128] %
// UNWRAP-LABEL: define{{.*}} void @uld(ppc_fp128 inreg %
void uld(ULD x) {}

// GPR-LABEL:    define{{.*}} void @uv([1 x i128] %
// UNWRAP-LABEL: define{{.*}} void @uv(<4 x float> inreg %
void uv(UV x) {}

// GPR-LABEL:    define{{.*}} void @su1(i32 %
// UNWRAP-LABEL: define{{.*}} void @su1(float inreg %
void su1(SU1 x) {}

// GPR-LABEL:    define{{.*}} void @us1(i32 %
// UNWRAP-LABEL: define{{.*}} void @us1(float inreg %
void us1(US1 x) {}

// GPR-LABEL:    define{{.*}} void @sau1(i32 %
// UNWRAP-LABEL: define{{.*}} void @sau1(float inreg %
void sau1(SAU1 x) {}

// A union of a vector is still quadword-aligned in the parameter save area.
// ALL-LABEL: define{{.*}} void @uva(
// ALL:       call ptr @llvm.ptrmask.p0.i64(ptr %{{.*}}, i64 -16)
void uva(int n, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, n);
  UV x = __builtin_va_arg(ap, UV);
  __builtin_va_end(ap);
}
