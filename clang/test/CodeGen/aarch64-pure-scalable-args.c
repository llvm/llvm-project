// RUN: %clang_cc1 -O3 -triple aarch64                                  -target-feature +sve -target-feature +sve2p1 -mvscale-min=1 -mvscale-max=1 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-AAPCS
// RUN: %clang_cc1 -O3 -triple arm64-apple-ios7.0 -target-abi darwinpcs -target-feature +sve -target-feature +sve2p1 -mvscale-min=1 -mvscale-max=1 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DARWIN
// RUN: %clang_cc1 -O3 -triple aarch64-linux-gnu                        -target-feature +sve -target-feature +sve2p1 -mvscale-min=1 -mvscale-max=1 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-AAPCS

// REQUIRES: aarch64-registered-target

#include <arm_sve.h>

typedef svfloat32_t fvec32 __attribute__((arm_sve_vector_bits(128)));
typedef svfloat64_t fvec64 __attribute__((arm_sve_vector_bits(128)));
typedef svbool_t bvec __attribute__((arm_sve_vector_bits(128)));
typedef svmfloat8_t mfvec8 __attribute__((arm_sve_vector_bits(128)));

typedef struct {
    float f[4];
} HFA;

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
// CHECK-AAPCS:      define dso_local void @test_argpass_simple(ptr nocapture noundef readonly %p)
// CHECK-AAPCS-NEXT: entry:
// CHECK-AAPCS-NEXT: %0 = load <2 x i8>, ptr %p, align 16
// CHECK-AAPCS-NEXT: %cast.scalable = tail call <vscale x 2 x i8> @llvm.vector.insert.nxv2i8.v2i8(<vscale x 2 x i8> undef, <2 x i8> %0, i64 0)
// CHECK-AAPCS-NEXT: %1 = bitcast <vscale x 2 x i8> %cast.scalable to <vscale x 16 x i1>
// CHECK-AAPCS-NEXT: %2 = getelementptr inbounds nuw i8, ptr %p, i64 16
// CHECK-AAPCS-NEXT: %3 = load <2 x double>, ptr %2, align 16
// CHECK-AAPCS-NEXT: %cast.scalable1 = tail call <vscale x 2 x double> @llvm.vector.insert.nxv2f64.v2f64(<vscale x 2 x double> undef, <2 x double> %3, i64 0)
// CHECK-AAPCS-NEXT: %4 = getelementptr inbounds nuw i8, ptr %p, i64 32
// CHECK-AAPCS-NEXT: %5 = load <4 x float>, ptr %4, align 16
// CHECK-AAPCS-NEXT: %cast.scalable2 = tail call <vscale x 4 x float> @llvm.vector.insert.nxv4f32.v4f32(<vscale x 4 x float> undef, <4 x float> %5, i64 0)
// CHECK-AAPCS-NEXT: %6 = getelementptr inbounds nuw i8, ptr %p, i64 48
// CHECK-AAPCS-NEXT: %7 = load <4 x float>, ptr %6, align 16
// CHECK-AAPCS-NEXT: %cast.scalable3 = tail call <vscale x 4 x float> @llvm.vector.insert.nxv4f32.v4f32(<vscale x 4 x float> undef, <4 x float> %7, i64 0)
// CHECK-AAPCS-NEXT: %8 = getelementptr inbounds nuw i8, ptr %p, i64 64
// CHECK-AAPCS-NEXT: %9 = load <16 x i8>, ptr %8, align 16
// CHECK-AAPCS-NEXT: %cast.scalable4 = tail call <vscale x 16 x i8> @llvm.vector.insert.nxv16i8.v16i8(<vscale x 16 x i8> undef, <16 x i8> %9, i64 0)
// CHECK-AAPCS-NEXT: %10 = getelementptr inbounds nuw i8, ptr %p, i64 80
// CHECK-AAPCS-NEXT: %11 = load <2 x i8>, ptr %10, align 16
// CHECK-AAPCS-NEXT: %cast.scalable5 = tail call <vscale x 2 x i8> @llvm.vector.insert.nxv2i8.v2i8(<vscale x 2 x i8> undef, <2 x i8> %11, i64 0)
// CHECK-AAPCS-NEXT: %12 = bitcast <vscale x 2 x i8> %cast.scalable5 to <vscale x 16 x i1>
// CHECK-AAPCS-NEXT: tail call void @argpass_simple_callee(<vscale x 16 x i1> %1, <vscale x 2 x double> %cast.scalable1, <vscale x 4 x float> %cast.scalable2, <vscale x 4 x float> %cast.scalable3, <vscale x 16 x i8> %cast.scalable4, <vscale x 16 x i1> %12)
// CHECK-AAPCS-NEXT: ret void

// CHECK-AAPCS:  declare void @argpass_simple_callee(<vscale x 16 x i1>, <vscale x 2 x double>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 16 x i8>, <vscale x 16 x i1>)
// CHECK-DARWIN: declare void @argpass_simple_callee(ptr noundef)

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
// CHECK-DARWIN: declare void @argpass_last_z_callee(double noundef, double noundef, double noundef, double noundef, ptr noundef)


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
// CHECK-DARWIN: declare void @argpass_last_z_tuple_callee(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, ptr noundef)


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
// CHECK-DARWIN: declare void @argpass_last_p_callee(<vscale x 16 x i1>, target("aarch64.svcount"), ptr noundef)


// Not enough Z-regs, push PST to memory and pass a pointer, Z-regs and
// P-regs still available for other arguments
//   u     -> z0
//   0.0   -> d1-d4
//   1     -> w0
//   *p    -> memory, address -> x1
//   2     -> w2
//   3.0   -> d5
//   true  -> p0
void test_argpass_no_z(PST *p, double dummy, svmfloat8_t u) {
    void argpass_no_z_callee(svmfloat8_t, double, double, double, double, int, PST, int, double, svbool_t);
    argpass_no_z_callee(u, .0, .0, .0, .0, 1, *p, 2, 3.0, svptrue_b64());
}
// CHECK: declare void @argpass_no_z_callee(<vscale x 16 x i8>, double noundef, double noundef, double noundef, double noundef, i32 noundef, ptr noundef, i32 noundef, double noundef, <vscale x 16 x i1>)


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
// CHECK: declare void @argpass_no_z_tuple_f64_callee(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, double noundef, i32 noundef, ptr noundef, i32 noundef, double noundef, <vscale x 16 x i1>)


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
// CHECK: declare void @argpass_no_z_tuple_mfp8_callee(<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, double noundef, i32 noundef, ptr noundef, i32 noundef, double noundef, <vscale x 16 x i1>)


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
// CHECK-AAPCS:  declare void @argpass_no_z_hfa_callee(double noundef, [4 x float] alignstack(8), i32 noundef, ptr noundef, i32 noundef, <vscale x 16 x i1>)
// CHECK-DARWIN: declare void @argpass_no_z_hfa_callee(double noundef, [4 x float], i32 noundef, ptr noundef, i32 noundef, <vscale x 16 x i1>)


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
// CHECK: declare void @argpass_no_p_callee(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>, i32 noundef, ptr noundef, i32 noundef, double noundef, <vscale x 16 x i1>)


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
// CHECK: declare void @argpass_no_p_tuple_callee(<vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>, i32 noundef, ptr noundef, i32 noundef, double noundef, <vscale x 16 x i1>)


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
// CHECK-AAPCS:  declare void @after_hfa_callee(double noundef, double noundef, double noundef, double noundef, double noundef, [4 x float] alignstack(8), ptr noundef, [4 x float] alignstack(8), <vscale x 16 x i1>)
// CHECK-DARWIN: declare void @after_hfa_callee(double noundef, double noundef, double noundef, double noundef, double noundef, [4 x float], ptr noundef, [4 x float], <vscale x 16 x i1>)

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
// CHECK-AAPCS:  declare void @small_pst_callee([2 x i64], double noundef, double noundef, double noundef, double noundef, double noundef, double noundef, double noundef, double noundef, double noundef, ptr noundef, double noundef)
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
// CHECK-DARWIN: define void @test_return(ptr dead_on_unwind noalias nocapture writable writeonly sret(%struct.PST) align 16 %agg.result, ptr nocapture noundef readonly %p)

// Corner case of 1-element aggregate
//   p->x -> q0
SmallPST test_return_small_pst(SmallPST *p) {
    return *p;
}
// CHECK-AAPCS:  define dso_local <vscale x 4 x float> @test_return_small_pst(ptr
// CHECK-DARWIN: define i128 @test_return_small_pst(ptr nocapture noundef readonly %p)


// Big PST, returned indirectly
//   *p -> *x8
BigPST test_return_big_pst(BigPST *p) {
    return *p;
}
// CHECK-AAPCS:  define dso_local void @test_return_big_pst(ptr dead_on_unwind noalias nocapture writable writeonly sret(%struct.BigPST) align 16 %agg.result, ptr nocapture noundef readonly %p)
// CHECK-DARWIN: define void @test_return_big_pst(ptr dead_on_unwind noalias nocapture writable writeonly sret(%struct.BigPST) align 16 %agg.result, ptr nocapture noundef readonly %p)

// Variadic arguments are unnamed, PST passed indirectly
//   0  -> x0
//   *p -> memory, address -> x1
void test_pass_variadic(PST *p) {
    void pass_variadic_callee(int n, ...);
    pass_variadic_callee(0, *p);
}
// CHECK: declare void @pass_variadic_callee(i32 noundef, ...)


// Test handling of a PST argument when passed in registers, from the callee side.
void argpass_callee_side(PST v) {
    void use(PST *p);
    use(&v);
}
// CHECK-AAPCS:      define dso_local void @argpass_callee_side(<vscale x 16 x i1> %0, <vscale x 2 x double> %.coerce1, <vscale x 4 x float> %.coerce3, <vscale x 4 x float> %.coerce5, <vscale x 16 x i8> %.coerce7, <vscale x 16 x i1> %1) local_unnamed_addr #0 {
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
// CHECK-AAPCS-NEXT:   call void @use(ptr noundef nonnull %v) #8
// CHECK-AAPCS-NEXT:   ret void
// CHECK-AAPCS-NEXT: }
