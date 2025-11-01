// RUN: %clang_cc1 -O3 -triple aarch64                                  -target-feature +sve -target-feature +sve2p1 -mvscale-min=1 -mvscale-max=1 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-AAPCS
// RUN: %clang_cc1 -O3 -triple arm64-apple-ios7.0 -target-abi darwinpcs -target-feature +sve -target-feature +sve2p1 -mvscale-min=1 -mvscale-max=1 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DARWIN
// RUN: %clang_cc1 -O3 -triple aarch64-linux-gnu                        -target-feature +sve -target-feature +sve2p1 -mvscale-min=1 -mvscale-max=1 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-AAPCS

// REQUIRES: aarch64-registered-target

#include <arm_neon.h>
#include <arm_sve.h>
#include <stdarg.h>

typedef svfloat32_t fvec32 __attribute__((arm_sve_vector_bits(128)));
typedef svfloat64_t fvec64 __attribute__((arm_sve_vector_bits(128)));
typedef svbool_t bvec __attribute__((arm_sve_vector_bits(128)));
typedef svmfloat8_t mfvec8 __attribute__((arm_sve_vector_bits(128)));

typedef struct {
    float f[4];
} HFA;

typedef struct {
    mfloat8x16_t f[4];
} HVA;

// Pure Scalable Type, needs 4 Z-regs, 2 P-regs
typedef struct {
     bvec a;
     fvec64 x;
     fvec32 y[2];
     mfvec8 z;
     bvec b;
} PST;

// Pure Scalable Type, 1 Z-reg
typedef struct {
    fvec32 x;
} SmallPST;

// Big PST, does not fit in registers.
typedef struct {
    struct {
        bvec a;
        fvec32 x[4];
    } u[2];
    fvec64 v;
} BigPST;

// A small aggregate type
typedef struct  {
    char data[16];
} SmallAgg;

// CHECK: %struct.PST = type { <2 x i8>, <2 x double>, [2 x <4 x float>], <16 x i8>, <2 x i8> }

// Test argument passing of Pure Scalable Types by examining the generated
// LLVM IR function declarations. A PST argument in C/C++ should map to:
//   a) an `ptr` argument, if passed indirectly through memory
//   b) a series of scalable vector arguments, if passed via registers

// Simple argument passing, PST expanded into registers.
//   a    -> p0
//   b    -> p1
//   x    -> q0
//   y[0] -> q1
//   y[1] -> q2
//   z    -> q3
void test_argpass_simple(PST *p) {
    void argpass_simple_callee(PST);
    argpass_simple_callee(*p);
}
// CHECK-AAPCS:      define dso_local void @test_argpass_simple(ptr noundef readonly captures(none) %p)
// CHECK-AAPCS-NEXT: entry:
// CHECK-AAPCS-NEXT: %0 = load <2 x i8>, ptr %p, align 16
// CHECK-AAPCS-NEXT: %cast.scalable = tail call <vscale x 2 x i8> @llvm.vector.insert.nxv2i8.v2i8(<vscale x 2 x i8> poison, <2 x i8> %0, i64 0)
// CHECK-AAPCS-NEXT: %1 = bitcast <vscale x 2 x i8> %cast.scalable to <vscale x 16 x i1>
// CHECK-AAPCS-NEXT: %2 = getelementptr inbounds nuw i8, ptr %p, i64 16
// CHECK-AAPCS-NEXT: %3 = load <2 x double>, ptr %2, align 16
// CHECK-AAPCS-NEXT: %cast.scalable1 = tail call <vscale x 2 x double> @llvm.vector.insert.nxv2f64.v2f64(<vscale x 2 x double> poison, <2 x double> %3, i64 0)
// CHECK-AAPCS-NEXT: %4 = getelementptr inbounds nuw i8, ptr %p, i64 32
// CHECK-AAPCS-NEXT: %5 = load <4 x float>, ptr %4, align 16
// CHECK-AAPCS-NEXT: %cast.scalable2 = tail call <vscale x 4 x float> @llvm.vector.insert.nxv4f32.v4f32(<vscale x 4 x float> poison, <4 x float> %5, i64 0)
// CHECK-AAPCS-NEXT: %6 = getelementptr inbounds nuw i8, ptr %p, i64 48
// CHECK-AAPCS-NEXT: %7 = load <4 x float>, ptr %6, align 16
// CHECK-AAPCS-NEXT: %cast.scalable3 = tail call <vscale x 4 x float> @llvm.vector.insert.nxv4f32.v4f32(<vscale x 4 x float> poison, <4 x float> %7, i64 0)
// CHECK-AAPCS-NEXT: %8 = getelementptr inbounds nuw i8, ptr %p, i64 64
// CHECK-AAPCS-NEXT: %9 = load <16 x i8>, ptr %8, align 16
// CHECK-AAPCS-NEXT: %cast.scalable4 = tail call <vscale x 16 x i8> @llvm.vector.insert.nxv16i8.v16i8(<vscale x 16 x i8> poison, <16 x i8> %9, i64 0)
// CHECK-AAPCS-NEXT: %10 = getelementptr inbounds nuw i8, ptr %p, i64 80
// CHECK-AAPCS-NEXT: %11 = load <2 x i8>, ptr %10, align 16
// CHECK-AAPCS-NEXT: %cast.scalable5 = tail call <vscale x 2 x i8> @llvm.vector.insert.nxv2i8.v2i8(<vscale x 2 x i8> poison, <2 x i8> %11, i64 0)
// CHECK-AAPCS-NEXT: %12 = bitcast <vscale x 2 x i8> %cast.scalable5 to <vscale x 16 x i1>
// CHECK-AAPCS-NEXT: tail call void @argpass_simple_callee(<vscale x 16 x i1> %1, <vscale x 2 x double> %cast.scalable1, <vscale x 4 x float> %cast.scalable2, <vscale x 4 x float> %cast.scalable3, <vscale x 16 x i8> %cast.scalable4, <vscale x 16 x i1> %12)
// CHECK-AAPCS-NEXT: ret void

// CHECK-AAPCS:  declare void @argpass_simple_callee(<vscale x 16 x i1>, <vscale x 2 x double>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 16 x i8>, <vscale x 16 x i1>)
// CHECK-DARWIN: declare void @argpass_simple_callee(ptr dead_on_return noundef)

// Boundary case of using the last available Z-reg, PST expanded.
//   0.0  -> d0-d3
//   a    -> p0
//   b    -> p1
//   x    -> q4
//   y[0] -> q5
//   y[1] -> q6
//   z    -> q7
void test_argpass_last_z(PST *p) {
    void argpass_last_z_callee(double, double, double, double, PST);
    argpass_last_z_callee(.0, .0, .0, .0, *p);
}
// CHECK-AAPCS:  declare void @argpass_last_z_callee(double noundef, double noundef, double noundef, double noundef, <vscale x 16 x i1>, <vscale x 2 x double>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 16 x i8>, <vscale x 16 x i1>)
// CHECK-DARWIN: declare void @argpass_last_z_callee(double noundef, double noundef, double noundef, double noundef, ptr dead_on_return noundef)


// Like the above, but using a tuple type to occupy some registers.
//   x    -> z0.d-z3.d
//   a    -> p0
//   b    -> p1
//   x    -> q4
//   y[0] -> q5
//   y[1] -> q6
//   z    -> q7
void test_argpass_last_z_tuple(PST *p, svfloat64x4_t x) {
  void argpass_last_z_tuple_callee(svfloat64x4_t, PST);
  argpass_last_z_tuple_callee(x, *p);
}
// CHECK-AAPCS:  declare void @argpass_last_z_tuple_callee(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 16 x i1>, <vscale x 2 x double>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 16 x i8>, <vscale x 16 x i1>)
// CHECK-DARWIN: declare void @argpass_last_z_tuple_callee(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, ptr dead_on_return noundef)


// Boundary case of using the last available P-reg, PST expanded.
//   false -> p0-p1
//   a     -> p2
//   b     -> p3
//   x     -> q0
//   y[0]  -> q1
//   y[1]  -> q2
//   z     -> q3
void test_argpass_last_p(PST *p) {
    void argpass_last_p_callee(svbool_t, svcount_t, PST);
    argpass_last_p_callee(svpfalse(), svpfalse_c(), *p);
}
// CHECK-AAPCS:  declare void @argpass_last_p_callee(<vscale x 16 x i1>, target("aarch64.svcount"), <vscale x 16 x i1>, <vscale x 2 x double>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 16 x i8>, <vscale x 16 x i1>)
// CHECK-DARWIN: declare void @argpass_last_p_callee(<vscale x 16 x i1>, target("aarch64.svcount"), ptr dead_on_return noundef)


// Not enough Z-regs, push PST to memory and pass a pointer, Z-regs and
// P-regs still available for other arguments
//   u     -> z0
//   v     -> q1
//   w     -> q2
//   0.0   -> d3-d4
//   1     -> w0
//   *p    -> memory, address -> x1
//   2     -> w2
//   3.0   -> d5
//   true  -> p0
void test_argpass_no_z(PST *p, double dummy, svmfloat8_t u, int8x16_t v, mfloat8x16_t w) {
    void argpass_no_z_callee(svmfloat8_t, int8x16_t, mfloat8x16_t, double, double, int, PST, int, double, svbool_t);
    argpass_no_z_callee(u, v, w, .0, .0, 1, *p, 2, 3.0, svptrue_b64());
}
// CHECK: declare void @argpass_no_z_callee(<vscale x 16 x i8>, <16 x i8> noundef, <16 x i8>, double noundef, double noundef, i32 noundef, ptr dead_on_return noundef, i32 noundef, double noundef, <vscale x 16 x i1>)


// Like the above, using a tuple to occupy some registers.
//   x     -> z0.d-z3.d
//   0.0   -> d4
//   1     -> w0
//   *p    -> memory, address -> x1
//   2     -> w2
//   3.0   -> d5
//   true  -> p0
void test_argpass_no_z_tuple_f64(PST *p, float dummy, svfloat64x4_t x) {
  void argpass_no_z_tuple_f64_callee(svfloat64x4_t, double, int, PST, int,
                                     double, svbool_t);
  argpass_no_z_tuple_f64_callee(x, .0, 1, *p, 2, 3.0, svptrue_b64());
}
// CHECK: declare void @argpass_no_z_tuple_f64_callee(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, double noundef, i32 noundef, ptr dead_on_return noundef, i32 noundef, double noundef, <vscale x 16 x i1>)


// Likewise, using a different tuple.
//   x     -> z0.d-z3.d
//   0.0   -> d4
//   1     -> w0
//   *p    -> memory, address -> x1
//   2     -> w2
//   3.0   -> d5
//   true  -> p0
void test_argpass_no_z_tuple_mfp8(PST *p, float dummy, svmfloat8x4_t x) {
  void argpass_no_z_tuple_mfp8_callee(svmfloat8x4_t, double, int, PST, int,
                                      double, svbool_t);
  argpass_no_z_tuple_mfp8_callee(x, .0, 1, *p, 2, 3.0, svptrue_b64());
}
// CHECK: declare void @argpass_no_z_tuple_mfp8_callee(<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, double noundef, i32 noundef, ptr dead_on_return noundef, i32 noundef, double noundef, <vscale x 16 x i1>)


// Not enough Z-regs (consumed by a HFA), PST passed indirectly
//   0.0  -> d0
//   *h   -> s1-s4
//   1    -> w0
//   *p   -> memory, address -> x1
//   p    -> x1
//   2    -> w2
//   true -> p0
void test_argpass_no_z_hfa(HFA *h, PST *p) {
    void argpass_no_z_hfa_callee(double, HFA, int, PST, int, svbool_t);
    argpass_no_z_hfa_callee(.0, *h, 1, *p, 2, svptrue_b64());
}
// CHECK-AAPCS:  declare void @argpass_no_z_hfa_callee(double noundef, [4 x float] alignstack(8), i32 noundef, ptr dead_on_return noundef, i32 noundef, <vscale x 16 x i1>)
// CHECK-DARWIN: declare void @argpass_no_z_hfa_callee(double noundef, [4 x float], i32 noundef, ptr dead_on_return noundef, i32 noundef, <vscale x 16 x i1>)

// Not enough Z-regs (consumed by a HVA), PST passed indirectly
//   0.0  -> d0
//   *h   -> s1-s4
//   1    -> w0
//   *p   -> memory, address -> x1
//   p    -> x1
//   2    -> w2
//   true -> p0
void test_argpass_no_z_hva(HVA *h, PST *p) {
    void argpass_no_z_hva_callee(double, HVA, int, PST, int, svbool_t);
    argpass_no_z_hva_callee(.0, *h, 1, *p, 2, svptrue_b64());
}
// CHECK-AAPCS:  declare void @argpass_no_z_hva_callee(double noundef, [4 x <16 x i8>] alignstack(16), i32 noundef, ptr dead_on_return noundef, i32 noundef, <vscale x 16 x i1>)
// CHECK-DARWIN: declare void @argpass_no_z_hva_callee(double noundef, [4 x <16 x i8>], i32 noundef, ptr dead_on_return noundef, i32 noundef, <vscale x 16 x i1>)

// Not enough P-regs, PST passed indirectly, Z-regs and P-regs still available.
//   true -> p0-p2
//   1    -> w0
//   *p   -> memory, address -> x1
//   2    -> w2
//   3.0  -> d0
//   true -> p3
void test_argpass_no_p(PST *p) {
    void argpass_no_p_callee(svbool_t, svbool_t, svbool_t, int, PST, int, double, svbool_t);
    argpass_no_p_callee(svptrue_b8(), svptrue_b16(), svptrue_b32(), 1, *p, 2, 3.0, svptrue_b64());
}
// CHECK: declare void @argpass_no_p_callee(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>, i32 noundef, ptr dead_on_return noundef, i32 noundef, double noundef, <vscale x 16 x i1>)


// Like above, using a tuple to occupy some registers.
// P-regs still available.
//   v    -> p0-p1
//   u    -> p2
//   1    -> w0
//   *p   -> memory, address -> x1
//   2    -> w2
//   3.0  -> d0
//   true -> p3
void test_argpass_no_p_tuple(PST *p, svbool_t u, svboolx2_t v) {
  void argpass_no_p_tuple_callee(svboolx2_t, svbool_t, int, PST, int, double,
                                 svbool_t);
  argpass_no_p_tuple_callee(v, u, 1, *p, 2, 3.0, svptrue_b64());
}
// CHECK: declare void @argpass_no_p_tuple_callee(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>, i32 noundef, ptr dead_on_return noundef, i32 noundef, double noundef, <vscale x 16 x i1>)


// HFAs go back-to-back to memory, afterwards Z-regs not available, PST passed indirectly.
//   0.0   -> d0-d3
//   *h    -> memory
//   *p    -> memory, address -> x0
//   *h    -> memory
//   false -> p0
void test_after_hfa(HFA *h, PST *p) {
    void after_hfa_callee(double, double, double, double, double, HFA, PST, HFA, svbool_t);
    after_hfa_callee(.0, .0, .0, .0, .0, *h, *p, *h, svpfalse());
}
// CHECK-AAPCS:  declare void @after_hfa_callee(double noundef, double noundef, double noundef, double noundef, double noundef, [4 x float] alignstack(8), ptr dead_on_return noundef, [4 x float] alignstack(8), <vscale x 16 x i1>)
// CHECK-DARWIN: declare void @after_hfa_callee(double noundef, double noundef, double noundef, double noundef, double noundef, [4 x float], ptr dead_on_return noundef, [4 x float], <vscale x 16 x i1>)

// Small PST, not enough registers, passed indirectly, unlike other small
// aggregates.
//   *s  -> x0-x1
//   0.0 -> d0-d7
//   *p  -> memory, address -> x2
//   1.0 -> memory
//   2.0 -> memory (next to the above)
void test_small_pst(SmallPST *p, SmallAgg *s) {
    void small_pst_callee(SmallAgg, double, double, double, double, double, double, double, double, double, SmallPST, double);
    small_pst_callee(*s, .0, .0, .0, .0, .0, .0, .0, .0, 1.0, *p, 2.0);
}
// CHECK-AAPCS:  declare void @small_pst_callee([2 x i64], double noundef, double noundef, double noundef, double noundef, double noundef, double noundef, double noundef, double noundef, double noundef, ptr dead_on_return noundef, double noundef)
// CHECK-DARWIN: declare void @small_pst_callee([2 x i64], double noundef, double noundef, double noundef, double noundef, double noundef, double noundef, double noundef, double noundef, double noundef, i128, double noundef)


// Simple return, PST expanded to registers
//   p->a    -> p0
//   p->x    -> q0
//   p->y[0] -> q1
//   p->y[1] -> q2
//   p->z    -> q3
//   p->b    -> p1
PST test_return(PST *p) {
    return *p;
}
// CHECK-AAPCS:  define dso_local <{ <vscale x 16 x i1>, <vscale x 2 x double>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 16 x i8>, <vscale x 16 x i1> }> @test_return(ptr
// CHECK-DARWIN: define void @test_return(ptr dead_on_unwind noalias writable writeonly sret(%struct.PST) align 16 captures(none) initializes((0, 96)) %agg.result, ptr noundef readonly captures(none) %p)

// Corner case of 1-element aggregate
//   p->x -> q0
SmallPST test_return_small_pst(SmallPST *p) {
    return *p;
}
// CHECK-AAPCS:  define dso_local <vscale x 4 x float> @test_return_small_pst(ptr
// CHECK-DARWIN: define i128 @test_return_small_pst(ptr noundef readonly captures(none) %p)


// Big PST, returned indirectly
//   *p -> *x8
BigPST test_return_big_pst(BigPST *p) {
    return *p;
}
// CHECK-AAPCS:  define dso_local void @test_return_big_pst(ptr dead_on_unwind noalias writable writeonly sret(%struct.BigPST) align 16 captures(none) initializes((0, 176)) %agg.result, ptr noundef readonly captures(none) %p)
// CHECK-DARWIN: define void @test_return_big_pst(ptr dead_on_unwind noalias writable writeonly sret(%struct.BigPST) align 16 captures(none) initializes((0, 176)) %agg.result, ptr noundef readonly captures(none) %p)

// Variadic arguments are unnamed, PST passed indirectly.
// (Passing SVE types to a variadic function currently unsupported by
// the AArch64 backend)
//   p->a    -> p0
//   p->x    -> q0
//   p->y[0] -> q1
//   p->y[1] -> q2
//   p->z    -> q3
//   p->b    -> p1
//   *q -> memory, address -> x1
void test_pass_variadic(PST *p, PST *q) {
    void pass_variadic_callee(PST, ...);
    pass_variadic_callee(*p, *q);
}
// CHECK-AAPCS: call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 16 dereferenceable(96) %byval-temp, ptr noundef nonnull align 16 dereferenceable(96) %q, i64 96, i1 false)
// CHECK-AAPCS: call void (<vscale x 16 x i1>, <vscale x 2 x double>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 16 x i8>, <vscale x 16 x i1>, ...) @pass_variadic_callee(<vscale x 16 x i1> %1, <vscale x 2 x double> %cast.scalable1, <vscale x 4 x float> %cast.scalable2, <vscale x 4 x float> %cast.scalable3, <vscale x 16 x i8> %cast.scalable4, <vscale x 16 x i1> %12, ptr dead_on_return noundef nonnull %byval-temp)

// CHECK-DARWIN: call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 16 dereferenceable(96) %byval-temp, ptr noundef nonnull align 16 dereferenceable(96) %p, i64 96, i1 false)
// CHECK-DARWIN: call void @llvm.lifetime.start.p0(ptr nonnull %byval-temp1)
// CHECK-DARWIN: call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 16 dereferenceable(96) %byval-temp1, ptr noundef nonnull align 16 dereferenceable(96) %q, i64 96, i1 false)
// CHECK-DARWIN: call void (ptr, ...) @pass_variadic_callee(ptr dead_on_return noundef nonnull %byval-temp, ptr dead_on_return noundef nonnull %byval-temp1)


// Test passing a small PST, still passed indirectly, despite being <= 128 bits
void test_small_pst_variadic(SmallPST *p) {
    void small_pst_variadic_callee(int, ...);
    small_pst_variadic_callee(0, *p);
}
// CHECK-AAPCS: call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 16 dereferenceable(16) %byval-temp, ptr noundef nonnull align 16 dereferenceable(16) %p, i64 16, i1 false)
// CHECK-AAPCS: call void (i32, ...) @small_pst_variadic_callee(i32 noundef 0, ptr dead_on_return noundef nonnull %byval-temp)

// CHECK-DARWIN: %0 = load i128, ptr %p, align 16
// CHECK-DARWIN: tail call void (i32, ...) @small_pst_variadic_callee(i32 noundef 0, i128 %0)

// Test handling of a PST argument when passed in registers, from the callee side.
void test_argpass_callee_side(PST v) {
    void use(PST *p);
    use(&v);
}
// CHECK-AAPCS:      define dso_local void @test_argpass_callee_side(<vscale x 16 x i1> %0, <vscale x 2 x double> %.coerce1, <vscale x 4 x float> %.coerce3, <vscale x 4 x float> %.coerce5, <vscale x 16 x i8> %.coerce7, <vscale x 16 x i1> %1)
// CHECK-AAPCS-NEXT: entry:
// CHECK-AAPCS-NEXT:   %v = alloca %struct.PST, align 16
// CHECK-AAPCS-NEXT:   %.coerce = bitcast <vscale x 16 x i1> %0 to <vscale x 2 x i8>
// CHECK-AAPCS-NEXT:   %cast.fixed = tail call <2 x i8> @llvm.vector.extract.v2i8.nxv2i8(<vscale x 2 x i8> %.coerce, i64 0)
// CHECK-AAPCS-NEXT:   store <2 x i8> %cast.fixed, ptr %v, align 16
// CHECK-AAPCS-NEXT:   %2 = getelementptr inbounds nuw i8, ptr %v, i64 16
// CHECK-AAPCS-NEXT:   %cast.fixed2 = tail call <2 x double> @llvm.vector.extract.v2f64.nxv2f64(<vscale x 2 x double> %.coerce1, i64 0)
// CHECK-AAPCS-NEXT:   store <2 x double> %cast.fixed2, ptr %2, align 16
// CHECK-AAPCS-NEXT:   %3 = getelementptr inbounds nuw i8, ptr %v, i64 32
// CHECK-AAPCS-NEXT:   %cast.fixed4 = tail call <4 x float> @llvm.vector.extract.v4f32.nxv4f32(<vscale x 4 x float> %.coerce3, i64 0)
// CHECK-AAPCS-NEXT:   store <4 x float> %cast.fixed4, ptr %3, align 16
// CHECK-AAPCS-NEXT:   %4 = getelementptr inbounds nuw i8, ptr %v, i64 48
// CHECK-AAPCS-NEXT:   %cast.fixed6 = tail call <4 x float> @llvm.vector.extract.v4f32.nxv4f32(<vscale x 4 x float> %.coerce5, i64 0)
// CHECK-AAPCS-NEXT:   store <4 x float> %cast.fixed6, ptr %4, align 16
// CHECK-AAPCS-NEXT:   %5 = getelementptr inbounds nuw i8, ptr %v, i64 64
// CHECK-AAPCS-NEXT:   %cast.fixed8 = tail call <16 x i8> @llvm.vector.extract.v16i8.nxv16i8(<vscale x 16 x i8> %.coerce7, i64 0)
// CHECK-AAPCS-NEXT:   store <16 x i8> %cast.fixed8, ptr %5, align 16
// CHECK-AAPCS-NEXT:   %6 = getelementptr inbounds nuw i8, ptr %v, i64 80
// CHECK-AAPCS-NEXT:   %.coerce9 = bitcast <vscale x 16 x i1> %1 to <vscale x 2 x i8>
// CHECK-AAPCS-NEXT:   %cast.fixed10 = tail call <2 x i8> @llvm.vector.extract.v2i8.nxv2i8(<vscale x 2 x i8> %.coerce9, i64 0)
// CHECK-AAPCS-NEXT:   store <2 x i8> %cast.fixed10, ptr %6, align 16
// CHECK-AAPCS-NEXT:   call void @use(ptr noundef nonnull %v)
// CHECK-AAPCS-NEXT:   ret void
// CHECK-AAPCS-NEXT: }

// Test va_arg operation
#ifdef __cplusplus
 extern "C"
#endif
void test_va_arg(int n, ...) {
     va_list ap;
     va_start(ap, n);  
     PST v = va_arg(ap, PST);
     va_end(ap);

     void use1(bvec, fvec32);
     use1(v.a, v.y[1]);
}
// CHECK-AAPCS: define dso_local void @test_va_arg(i32 noundef %n, ...)
// CHECK-AAPCS-NEXT: entry:
// CHECK-AAPCS-NEXT:   %ap = alloca %struct.__va_list, align 8
// CHECK-AAPCS-NEXT:   call void @llvm.lifetime.start.p0(ptr nonnull %ap)
// CHECK-AAPCS-NEXT:   call void @llvm.va_start.p0(ptr nonnull %ap)
// CHECK-AAPCS-NEXT:   %gr_offs_p = getelementptr inbounds nuw i8, ptr %ap, i64 24
// CHECK-AAPCS-NEXT:   %gr_offs = load i32, ptr %gr_offs_p, align 8
// CHECK-AAPCS-NEXT:   %0 = icmp sgt i32 %gr_offs, -1
// CHECK-AAPCS-NEXT:   br i1 %0, label %vaarg.on_stack, label %vaarg.maybe_reg
// CHECK-AAPCS-EMPTY:
// CHECK-AAPCS-NEXT: vaarg.maybe_reg:                                  ; preds = %entry

// Increment by 8, size of the pointer to the argument value, not size of the argument value itself.

// CHECK-AAPCS-NEXT:   %new_reg_offs = add nsw i32 %gr_offs, 8
// CHECK-AAPCS-NEXT:   store i32 %new_reg_offs, ptr %gr_offs_p, align 8
// CHECK-AAPCS-NEXT:   %inreg = icmp samesign ult i32 %gr_offs, -7
// CHECK-AAPCS-NEXT:   br i1 %inreg, label %vaarg.in_reg, label %vaarg.on_stack
// CHECK-AAPCS-EMPTY:
// CHECK-AAPCS-NEXT: vaarg.in_reg:                                     ; preds = %vaarg.maybe_reg
// CHECK-AAPCS-NEXT:   %reg_top_p = getelementptr inbounds nuw i8, ptr %ap, i64 8
// CHECK-AAPCS-NEXT:   %reg_top = load ptr, ptr %reg_top_p, align 8
// CHECK-AAPCS-NEXT:   %1 = sext i32 %gr_offs to i64
// CHECK-AAPCS-NEXT:   %2 = getelementptr inbounds i8, ptr %reg_top, i64 %1
// CHECK-AAPCS-NEXT:   br label %vaarg.end
// CHECK-AAPCS-EMPTY:
// CHECK-AAPCS-NEXT: vaarg.on_stack:                                   ; preds = %vaarg.maybe_reg, %entry
// CHECK-AAPCS-NEXT:   %stack = load ptr, ptr %ap, align 8
// CHECK-AAPCS-NEXT:   %new_stack = getelementptr inbounds nuw i8, ptr %stack, i64 8
// CHECK-AAPCS-NEXT:   store ptr %new_stack, ptr %ap, align 8
// CHECK-AAPCS-NEXT:   br label %vaarg.end
// CHECK-AAPCS-EMPTY:
// CHECK-AAPCS-NEXT: vaarg.end:                                        ; preds = %vaarg.on_stack, %vaarg.in_reg
// CHECK-AAPCS-NEXT:   %vaargs.addr = phi ptr [ %2, %vaarg.in_reg ], [ %stack, %vaarg.on_stack ]

// Extra indirection, for a composite passed indirectly.
// CHECK-AAPCS-NEXT:   %vaarg.addr = load ptr, ptr %vaargs.addr, align 8

// CHECK-AAPCS-NEXT:   %v.sroa.0.0.copyload = load <2 x i8>, ptr %vaarg.addr, align 16
// CHECK-AAPCS-NEXT:   %v.sroa.43.0.vaarg.addr.sroa_idx = getelementptr inbounds nuw i8, ptr %vaarg.addr, i64 48
// CHECK-AAPCS-NEXT:   %v.sroa.43.0.copyload = load <4 x float>, ptr %v.sroa.43.0.vaarg.addr.sroa_idx, align 16
// CHECK-AAPCS-NEXT:   call void @llvm.va_end.p0(ptr nonnull %ap)
// CHECK-AAPCS-NEXT:   %cast.scalable = call <vscale x 2 x i8> @llvm.vector.insert.nxv2i8.v2i8(<vscale x 2 x i8> poison, <2 x i8> %v.sroa.0.0.copyload, i64 0)
// CHECK-AAPCS-NEXT:   %3 = bitcast <vscale x 2 x i8> %cast.scalable to <vscale x 16 x i1>
// CHECK-AAPCS-NEXT:   %cast.scalable2 = call <vscale x 4 x float> @llvm.vector.insert.nxv4f32.v4f32(<vscale x 4 x float> poison, <4 x float> %v.sroa.43.0.copyload, i64 0)
// CHECK-AAPCS-NEXT:   call void @use1(<vscale x 16 x i1> noundef %3, <vscale x 4 x float> noundef %cast.scalable2)
// CHECK-AAPCS-NEXT:   call void @llvm.lifetime.end.p0(ptr nonnull %ap)
// CHECK-AAPCS-NEXT:   ret void
// CHECK-AAPCS-NEXT: }

// CHECK-DARWIN: define void @test_va_arg(i32 noundef %n, ...)
// CHECK-DARWIN-NEXT: entry:
// CHECK-DARWIN-NEXT:   %ap = alloca ptr, align 8
// CHECK-DARWIN-NEXT:   call void @llvm.lifetime.start.p0(ptr nonnull %ap)
// CHECK-DARWIN-NEXT:   call void @llvm.va_start.p0(ptr nonnull %ap)
// CHECK-DARWIN-NEXT:   %argp.cur = load ptr, ptr %ap, align 8
// CHECK-DARWIN-NEXT:   %argp.next = getelementptr inbounds nuw i8, ptr %argp.cur, i64 8
// CHECK-DARWIN-NEXT:   store ptr %argp.next, ptr %ap, align 8
// CHECK-DARWIN-NEXT:   %0 = load ptr, ptr %argp.cur, align 8
// CHECK-DARWIN-NEXT:   %v.sroa.0.0.copyload = load <2 x i8>, ptr %0, align 16
// CHECK-DARWIN-NEXT:   %v.sroa.43.0..sroa_idx = getelementptr inbounds nuw i8, ptr %0, i64 48
// CHECK-DARWIN-NEXT:   %v.sroa.43.0.copyload = load <4 x float>, ptr %v.sroa.43.0..sroa_idx, align 16
// CHECK-DARWIN-NEXT:   call void @llvm.va_end.p0(ptr nonnull %ap)
// CHECK-DARWIN-NEXT:   %cast.scalable = call <vscale x 2 x i8> @llvm.vector.insert.nxv2i8.v2i8(<vscale x 2 x i8> poison, <2 x i8> %v.sroa.0.0.copyload, i64 0)
// CHECK-DARWIN-NEXT:   %1 = bitcast <vscale x 2 x i8> %cast.scalable to <vscale x 16 x i1>
// CHECK-DARWIN-NEXT:   %cast.scalable2 = call <vscale x 4 x float> @llvm.vector.insert.nxv4f32.v4f32(<vscale x 4 x float> poison, <4 x float> %v.sroa.43.0.copyload, i64 0)
// CHECK-DARWIN-NEXT:   call void @use1(<vscale x 16 x i1> noundef %1, <vscale x 4 x float> noundef %cast.scalable2)
// CHECK-DARWIN-NEXT:   call void @llvm.lifetime.end.p0(ptr nonnull %ap)
// CHECK-DARWIN-NEXT:   ret void
// CHECK-DARWIN-NEXT: }

// Regression test for incorrect passing of SVE vector tuples
// The whole `y` need to be passed indirectly.
void test_tuple_reg_count(svfloat32_t x, svfloat32x2_t y) {
  void test_tuple_reg_count_callee(svfloat32_t, svfloat32_t, svfloat32_t, svfloat32_t,
                                   svfloat32_t, svfloat32_t, svfloat32_t, svfloat32x2_t);
  test_tuple_reg_count_callee(x, x, x, x, x, x, x, y);
}
// CHECK-AAPCS: declare void @test_tuple_reg_count_callee(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, ptr dead_on_return noundef)
// CHECK-DARWIN: declare void @test_tuple_reg_count_callee(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>)

// Regression test for incorrect passing of SVE vector tuples
// The whole `y` need to be passed indirectly.
void test_tuple_reg_count_bool(svboolx4_t x, svboolx4_t y) {
  void test_tuple_reg_count_bool_callee(svboolx4_t, svboolx4_t);
  test_tuple_reg_count_bool_callee(x, y);
}
// CHECK-AAPCS:  declare void @test_tuple_reg_count_bool_callee(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>, ptr dead_on_return noundef)
// CHECK-DARWIN: declare void @test_tuple_reg_count_bool_callee(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>)
