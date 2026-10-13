// Verify that a riscv_rvv_vector_bits typedef keeps the LMUL of the sizeless
// RVV type it was written against, rather than having it re-derived from VLEN
// and the declared lane count.
//
// A typedef may declare fewer lanes than
// VLEN * LMUL, so NumElements / vscale under-estimates the LMUL. Two typedefs
// with the same element type and lane count but different LMULs must map to
// different scalable vector types.
//
// VLEN=128 (vscale=2): vint32mf2_t holds 2 i32 lanes at full size,
// vint32m1_t 4, vint32m2_t 8, vint32m4_t 16.

// RUN: %clang_cc1 -triple riscv64-none-linux-gnu \
// RUN:   -target-feature +f -target-feature +d -target-feature +zve64d \
// RUN:   -mvscale-min=2 -mvscale-max=2 \
// RUN:   -O1 -emit-llvm -o - %s | FileCheck %s

// REQUIRES: riscv-registered-target

#include <riscv_vector.h>

// All of these hold 2 i32 lanes; only the LMUL differs.
typedef vint32mf2_t mf2_64 __attribute__((riscv_rvv_vector_bits(64)));
typedef vint32m1_t  m1_64  __attribute__((riscv_rvv_vector_bits(64)));
typedef vint32m2_t  m2_64  __attribute__((riscv_rvv_vector_bits(64)));
typedef vint32m4_t  m4_64  __attribute__((riscv_rvv_vector_bits(64)));

// A single lane out of an LMUL=1 group.
typedef vint32m1_t m1_32 __attribute__((riscv_rvv_vector_bits(32)));

// Other element sizes.
typedef vint8m1_t    i8m1_32   __attribute__((riscv_rvv_vector_bits(32)));
typedef vint8mf4_t   i8mf4_16  __attribute__((riscv_rvv_vector_bits(16)));
typedef vfloat32m1_t f32m1_64  __attribute__((riscv_rvv_vector_bits(64)));
typedef vfloat64m1_t f64m1_64  __attribute__((riscv_rvv_vector_bits(64)));

//===----------------------------------------------------------------------===//
// Same lane count, different LMUL => different ABI types.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: define dso_local noundef <vscale x 1 x i32> @id_mf2_64(
// CHECK-SAME:    <vscale x 1 x i32> {{.*}}%x.coerce)
mf2_64 id_mf2_64(mf2_64 x) { return x; }

// CHECK-LABEL: define dso_local {{.*}}<vscale x 2 x i32> @id_m1_64(
// CHECK-SAME:    <vscale x 2 x i32> {{.*}}%x.coerce)
m1_64 id_m1_64(m1_64 x) { return x; }

// CHECK-LABEL: define dso_local {{.*}}<vscale x 4 x i32> @id_m2_64(
// CHECK-SAME:    <vscale x 4 x i32> {{.*}}%x.coerce)
m2_64 id_m2_64(m2_64 x) { return x; }

// Two lanes of an LMUL=4 type still occupy the whole LMUL=4 group.
// CHECK-LABEL: define dso_local {{.*}}<vscale x 8 x i32> @id_m4_64(
// CHECK-SAME:    <vscale x 8 x i32> {{.*}}%x.coerce)
m4_64 id_m4_64(m4_64 x) { return x; }

// One lane still occupies a whole LMUL=1 group.
// CHECK-LABEL: define dso_local {{.*}}<vscale x 2 x i32> @id_m1_32(
// CHECK-SAME:    <vscale x 2 x i32> {{.*}}%x.coerce)
m1_32 id_m1_32(m1_32 x) { return x; }

//===----------------------------------------------------------------------===//
// Other element sizes.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: define dso_local {{.*}}<vscale x 8 x i8> @id_i8m1_32(
// CHECK-SAME:    <vscale x 8 x i8> {{.*}}%x.coerce)
i8m1_32 id_i8m1_32(i8m1_32 x) { return x; }

// CHECK-LABEL: define dso_local {{.*}}<vscale x 2 x i8> @id_i8mf4_16(
// CHECK-SAME:    <vscale x 2 x i8> {{.*}}%x.coerce)
i8mf4_16 id_i8mf4_16(i8mf4_16 x) { return x; }

// CHECK-LABEL: define dso_local {{.*}}<vscale x 2 x float> @id_f32m1_64(
// CHECK-SAME:    <vscale x 2 x float> {{.*}}%x.coerce)
f32m1_64 id_f32m1_64(f32m1_64 x) { return x; }

// CHECK-LABEL: define dso_local {{.*}}<vscale x 1 x double> @id_f64m1_64(
// CHECK-SAME:    <vscale x 1 x double> {{.*}}%x.coerce)
f64m1_64 id_f64m1_64(f64m1_64 x) { return x; }

//===----------------------------------------------------------------------===//
// Calling RVV intrinsics with these typedefs. Because the ABI type already
// matches the LMUL the intrinsic expects, the argument flows straight into the
// intrinsic with no llvm.vector.insert / llvm.vector.extract in between.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: define dso_local {{.*}}<vscale x 2 x i32> @add_m1(
// CHECK-SAME:    <vscale x 2 x i32> {{.*}}%a.coerce, <vscale x 2 x i32> {{.*}}%b.coerce)
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[R:%.*]] = {{.*}}call <vscale x 2 x i32> @llvm.riscv.vadd.nxv2i32.nxv2i32.i64(<vscale x 2 x i32> poison, <vscale x 2 x i32> %a.coerce, <vscale x 2 x i32> %b.coerce, i64 2)
// CHECK-NEXT:    ret <vscale x 2 x i32> [[R]]
m1_64 add_m1(m1_64 a, m1_64 b) { return __riscv_vadd_vv_i32m1(a, b, 2); }

// Same lane count as add_m1, but the mf2 typedef selects the mf2 intrinsic.
// CHECK-LABEL: define dso_local <vscale x 1 x i32> @add_mf2(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[R:%.*]] = {{.*}}call <vscale x 1 x i32> @llvm.riscv.vadd.nxv1i32.nxv1i32.i64(<vscale x 1 x i32> poison, <vscale x 1 x i32> %a.coerce, <vscale x 1 x i32> %b.coerce, i64 2)
// CHECK-NEXT:    ret <vscale x 1 x i32> [[R]]
mf2_64 add_mf2(mf2_64 a, mf2_64 b) { return __riscv_vadd_vv_i32mf2(a, b, 2); }

// CHECK-LABEL: define dso_local {{.*}}<vscale x 4 x i32> @add_m2(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[R:%.*]] = {{.*}}call <vscale x 4 x i32> @llvm.riscv.vadd.nxv4i32.nxv4i32.i64(<vscale x 4 x i32> poison, <vscale x 4 x i32> %a.coerce, <vscale x 4 x i32> %b.coerce, i64 2)
// CHECK-NEXT:    ret <vscale x 4 x i32> [[R]]
m2_64 add_m2(m2_64 a, m2_64 b) { return __riscv_vadd_vv_i32m2(a, b, 2); }

// Fractional LMUL with a narrow element type.
// CHECK-LABEL: define dso_local {{.*}}<vscale x 2 x i8> @add_i8mf4(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[R:%.*]] = {{.*}}call <vscale x 2 x i8> @llvm.riscv.vadd.nxv2i8.nxv2i8.i64(<vscale x 2 x i8> poison, <vscale x 2 x i8> %a.coerce, <vscale x 2 x i8> %b.coerce, i64 2)
// CHECK-NEXT:    ret <vscale x 2 x i8> [[R]]
i8mf4_16 add_i8mf4(i8mf4_16 a, i8mf4_16 b) {
  return __riscv_vadd_vv_i8mf4(a, b, 2);
}

// Floating point.
// CHECK-LABEL: define dso_local {{.*}}<vscale x 2 x float> @fadd_m1(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[R:%.*]] = {{.*}}call <vscale x 2 x float> @llvm.riscv.vfadd.nxv2f32.nxv2f32.i64(<vscale x 2 x float> poison, <vscale x 2 x float> %a.coerce, <vscale x 2 x float> %b.coerce, i64 7, i64 2)
// CHECK-NEXT:    ret <vscale x 2 x float> [[R]]
f32m1_64 fadd_m1(f32m1_64 a, f32m1_64 b) {
  return __riscv_vfadd_vv_f32m1(a, b, 2);
}

// Load returning into the shorter typedef.
// CHECK-LABEL: define dso_local {{.*}}<vscale x 2 x i32> @load_m1(
// CHECK:         [[R:%.*]] = {{.*}}call <vscale x 2 x i32> @llvm.riscv.vle.nxv2i32.p0.i64(<vscale x 2 x i32> poison, ptr %p, i64 2)
// CHECK-NEXT:    ret <vscale x 2 x i32> [[R]]
m1_64 load_m1(const int *p) { return __riscv_vle32_v_i32m1(p, 2); }

// Store taking the shorter typedef.
// CHECK-LABEL: define dso_local void @store_m1(
// CHECK:         call void @llvm.riscv.vse.nxv2i32.p0.i64(<vscale x 2 x i32> %v.coerce, ptr %p, i64 2)
void store_m1(int *p, m1_64 v) { __riscv_vse32_v_i32m1(p, v, 2); }

// Mask-producing and masked intrinsics: the mask LMUL follows the data LMUL.
// CHECK-LABEL: define dso_local {{.*}}<vscale x 2 x i32> @masked_add_m1(
// CHECK:         [[M:%.*]] = {{.*}}call <vscale x 2 x i1> @llvm.riscv.vmseq.nxv2i32.nxv2i32.i64(<vscale x 2 x i32> %a.coerce, <vscale x 2 x i32> %b.coerce, i64 2)
// CHECK-NEXT:    [[R:%.*]] = {{.*}}call <vscale x 2 x i32> @llvm.riscv.vadd.mask.nxv2i32.nxv2i32.i64(<vscale x 2 x i32> poison, <vscale x 2 x i32> %a.coerce, <vscale x 2 x i32> %b.coerce, <vscale x 2 x i1> [[M]], i64 2, i64 3)
// CHECK-NEXT:    ret <vscale x 2 x i32> [[R]]
m1_64 masked_add_m1(m1_64 a, m1_64 b) {
  vbool32_t m = __riscv_vmseq_vv_i32m1_b32(a, b, 2);
  return __riscv_vadd_vv_i32m1_m(m, a, b, 2);
}

//===----------------------------------------------------------------------===//
// Structs. The number of vector registers a field needs follows its LMUL, not
// the number of lanes it declares.
//===----------------------------------------------------------------------===//

struct s1_m1 { m1_64 a; };
// CHECK-LABEL: define dso_local void @st1_m1(
// CHECK-SAME:    <vscale x 2 x i32>
void st1_m1(struct s1_m1 s) {}

struct s2_m1 { m1_64 a, b; };
// CHECK-LABEL: define dso_local void @st2_m1(
// CHECK-SAME:    target("riscv.vector.tuple", <vscale x 8 x i8>, 2)
void st2_m1(struct s2_m1 s) {}

struct s2_mf2 { mf2_64 a, b; };
// CHECK-LABEL: define dso_local void @st2_mf2(
// CHECK-SAME:    target("riscv.vector.tuple", <vscale x 4 x i8>, 2)
void st2_mf2(struct s2_mf2 s) {}

// 2 * LMUL4 = 8 registers, right at the limit.
struct s2_m4 { m4_64 a, b; };
// CHECK-LABEL: define dso_local void @st2_m4(
// CHECK-SAME:    target("riscv.vector.tuple", <vscale x 32 x i8>, 2)
void st2_m4(struct s2_m4 s) {}

// 3 * LMUL4 = 12 registers, over the limit, so the struct is passed
// indirectly. Deriving the LMUL from the lane count would have made this look
// like 3 * LMUL(1/2) and wrongly passed it in registers.
struct s3_m4 { m4_64 a, b, c; };
// CHECK-LABEL: define dso_local void @st3_m4(
// CHECK-SAME:    ptr
void st3_m4(struct s3_m4 s) {}
