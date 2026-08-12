// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-CIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-OGCG --input-file=%t.ll %s

// Anonymous record aliases are numbered in the order they are printed, so
// capture each one rather than naming it.
// CIR-DAG: ![[X87PAIR:rec_anon_struct[0-9]*]] = !cir.struct<{!cir.f80, !cir.f80}>
// CIR-DAG: ![[I64PAIR:rec_anon_struct[0-9]*]] = !cir.struct<{!u64i, !u64i}>
// CIR-DAG: ![[F64PAIR:rec_anon_struct[0-9]*]] = !cir.struct<{!cir.double, !cir.double}>
// CIR-DAG: ![[F32X2PAIR:rec_anon_struct[0-9]*]] = !cir.struct<{!cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>}>

typedef struct { int x; int y; } Pair2;
typedef struct { long a; long b; } Pair16;
typedef struct { long a, b, c, d; } Big;
typedef struct { long a; double b; } IntSSE;
typedef struct { double a; double b; } SSE2;
typedef struct { } Empty;
typedef union { int i; float f; } UIntFloat;
typedef union { float f; float g; } UFloats;
typedef union { int i; char c[8]; } UNarrowStorage;
typedef union { char c[32]; } UBig;
typedef union { char c[32]; } __attribute__((aligned(32))) UBigOverAligned;

// Narrow signed integer sign-extended in a register.
signed char ext_schar(signed char c) { return c; }

// CIR: cir.func {{.*}}@ext_schar(%arg0: !s8i {{.*}}llvm.signext{{.*}}) -> (!s8i {{.*}}llvm.signext
// LLVM: define dso_local signext i8 @ext_schar(i8 noundef signext %{{.+}})

// Narrow unsigned integer zero-extended in a register.
unsigned char ext_uchar(unsigned char c) { return c; }

// CIR: cir.func {{.*}}@ext_uchar(%arg0: !u8i {{.*}}llvm.zeroext{{.*}}) -> (!u8i {{.*}}llvm.zeroext
// LLVM: define dso_local zeroext i8 @ext_uchar(i8 noundef zeroext %{{.+}})

// Floating-point scalar passed/returned in an SSE register.
double sse_double(double d) { return d; }

// CIR: cir.func {{.*}}@sse_double(%arg0: !cir.double {{.*}}) -> !cir.double
// LLVM: define dso_local double @sse_double(double noundef %{{.+}})

// Two-int struct returned in a single INTEGER eightbyte -> i64.
Pair2 ret_pair2(int a) { Pair2 p = {a, a}; return p; }

// CIR: cir.func {{.*}}@ret_pair2(%arg0: !s32i {{.*}}) -> !u64i
// LLVM: define dso_local i64 @ret_pair2(i32 noundef %{{.+}})

// 16-byte struct flattened into two integer registers.
void take_pair16(Pair16 p) { (void)p; }

// CIR: cir.func {{.*}}@take_pair16(%arg0: !s64i{{.*}}, %arg1: !s64i{{.*}})
// LLVM: define dso_local void @take_pair16(i64 %{{.+}}, i64 %{{.+}})

// Struct split into one INTEGER and one SSE eightbyte.
void take_int_sse(IntSSE s) { (void)s; }

// CIR: cir.func {{.*}}@take_int_sse(%arg0: !s64i{{.*}}, %arg1: !cir.double{{.*}})
// LLVM: define dso_local void @take_int_sse(i64 %{{.+}}, double %{{.+}})

// Struct split into two SSE eightbytes.
void take_sse2(SSE2 s) { (void)s; }

// CIR: cir.func {{.*}}@take_sse2(%arg0: !cir.double{{.*}}, %arg1: !cir.double{{.*}})
// LLVM: define dso_local void @take_sse2(double %{{.+}}, double %{{.+}})

// Empty struct argument is ignored -- dropped from the signature entirely.
void take_empty(Empty e) { (void)e; }

// CIR: cir.func {{.*}}@take_empty()
// LLVM: define dso_local void @take_empty()

// Empty struct return is ignored -- the function returns void.
Empty ret_empty(void) { Empty e; return e; }

// CIR: cir.func {{.*}}@ret_empty()
// LLVM: define dso_local void @ret_empty()

// Large struct returned indirectly via sret.
Big ret_big(void) { Big b = {1, 2, 3, 4}; return b; }

// CIR: cir.func {{.*}}@ret_big(%arg0: !cir.ptr<!rec_Big> {{.*}}llvm.sret = !rec_Big{{.*}})
// LLVM: define dso_local void @ret_big(ptr dead_on_unwind noalias writable sret(%struct.Big) align 8 %{{.+}})

// Large struct passed byval.  CIR also emits noalias on byval; OGCG only does
// so under -fpass-by-value-is-noalias.
void take_big(Big b) { (void)b; }

// CIR: cir.func {{.*}}@take_big(%arg0: !cir.ptr<!rec_Big> {{.*}}llvm.byval = !rec_Big{{.*}})
// LLVM-CIR: define dso_local void @take_big(ptr noalias noundef byval(%struct.Big) align 8 %{{.+}})
// LLVM-OGCG: define dso_local void @take_big(ptr noundef byval(%struct.Big) align 8 %{{.+}})

// Union members all start at offset zero, so a 4-byte union takes one INTEGER
// eightbyte and coerces to i32.
void take_union(UIntFloat u) { (void)u; }

// CIR: cir.func {{.*}}@take_union(%arg0: !s32i{{.*}})
// LLVM: define dso_local void @take_union(i32 %{{.+}})

// A union of floats classifies SSE, so it coerces to a float register.
void take_union_floats(UFloats u) { (void)u; }

// CIR: cir.func {{.*}}@take_union_floats(%arg0: !cir.float{{.*}})
// LLVM: define dso_local void @take_union_floats(float %{{.+}})

// The union's highest-aligned member is the 4-byte int, but its size comes
// from the 8-byte array, and the eightbyte is sized from the union.
void take_union_narrow_storage(UNarrowStorage u) { (void)u; }

// CIR: cir.func {{.*}}@take_union_narrow_storage(%arg0: !u64i{{.*}})
// LLVM: define dso_local void @take_union_narrow_storage(i64 %{{.+}})

// A coerced union return round-trips through the coercion type.
UIntFloat ret_union(int a) { UIntFloat u; u.i = a; return u; }

// CIR: cir.func {{.*}}@ret_union(%arg0: !s32i {{.*}}) -> !s32i
// LLVM: define dso_local i32 @ret_union(i32 noundef %{{.+}})

// A union too large for registers is passed byval, with the same noalias
// divergence as a large struct.
void take_union_big(UBig u) { (void)u; }

// CIR: cir.func {{.*}}@take_union_big(%arg0: !cir.ptr<!rec_UBig> {{.*}}llvm.byval = !rec_UBig{{.*}})
// LLVM-CIR: define dso_local void @take_union_big(ptr noalias noundef byval(%union.UBig) align 8 %{{.+}})
// LLVM-OGCG: define dso_local void @take_union_big(ptr noundef byval(%union.UBig) align 8 %{{.+}})

// The byval alignment follows the union's declared alignment, not the alignment
// its members imply, which is 1 here.
void take_union_big_over_aligned(UBigOverAligned u) { (void)u; }

// CIR: cir.func {{.*}}@take_union_big_over_aligned(%arg0: !cir.ptr<!rec_UBigOverAligned> {{.*}}llvm.align = 32 : i64{{.*}}llvm.byval = !rec_UBigOverAligned{{.*}})
// LLVM-CIR: define dso_local void @take_union_big_over_aligned(ptr noalias noundef byval(%union.UBigOverAligned) align 32 %{{.+}})
// LLVM-OGCG: define dso_local void @take_union_big_over_aligned(ptr noundef byval(%union.UBigOverAligned) align 32 %{{.+}})

void call_union(UIntFloat u) { take_union(u); }

// CIR: cir.func {{.*}}@call_union(%arg0: !s32i
// CIR:   cir.call @take_union(%{{.+}}) : (!s32i) -> ()
// LLVM: define dso_local void @call_union(i32 %{{.+}})
// LLVM:   call void @take_union(i32 %{{.+}})

void call_union_big_over_aligned(UBigOverAligned u) {
  take_union_big_over_aligned(u);
}

// CIR: cir.func {{.*}}@call_union_big_over_aligned(%arg0: !cir.ptr<!rec_UBigOverAligned> {{.*}}llvm.align = 32 : i64{{.*}})
// CIR:   cir.call @take_union_big_over_aligned(%{{.+}}) : (!cir.ptr<!rec_UBigOverAligned> {{.*}}llvm.align = 32 : i64{{.*}}) -> ()
// LLVM-CIR: define dso_local void @call_union_big_over_aligned(ptr noalias noundef byval(%union.UBigOverAligned) align 32 %{{.+}})
// LLVM-CIR:   alloca %union.UBigOverAligned, align 32
// LLVM-CIR:   call void @take_union_big_over_aligned(ptr noalias noundef byval(%union.UBigOverAligned) align 32 %{{.+}})
// LLVM-OGCG: define dso_local void @call_union_big_over_aligned(ptr noundef byval(%union.UBigOverAligned) align 32 %{{.+}})
// LLVM-OGCG:   call void @take_union_big_over_aligned(ptr noundef byval(%union.UBigOverAligned) align 32 %{{.+}})

// The declared alignment reaches the sret slot of an indirect return too, not
// just a byval argument.
UBigOverAligned ret_union_big_over_aligned(void);
void call_ret_union_big_over_aligned(void) { (void)ret_union_big_over_aligned(); }

// CIR: cir.func {{.*}}@ret_union_big_over_aligned(!cir.ptr<!rec_UBigOverAligned> {{.*}}llvm.align = 32 : i64{{.*}}llvm.sret = !rec_UBigOverAligned{{.*}})
// LLVM: declare void @ret_union_big_over_aligned(ptr dead_on_unwind writable sret(%union.UBigOverAligned) align 32)

// The same declared-alignment source feeds an over-aligned struct, since
// mapCIRType's alignment lookup is on the shared record path, not a
// union-specific one.
typedef struct { char c[32]; } __attribute__((aligned(32))) SOverAligned;
void take_struct_over_aligned(SOverAligned s) { (void)s; }

// CIR: cir.func {{.*}}@take_struct_over_aligned(%arg0: !cir.ptr<!rec_SOverAligned> {{.*}}llvm.align = 32 : i64{{.*}}llvm.byval = !rec_SOverAligned{{.*}})
// LLVM-CIR: define dso_local void @take_struct_over_aligned(ptr noalias noundef byval(%struct.SOverAligned) align 32 %{{.+}})
// LLVM-OGCG: define dso_local void @take_struct_over_aligned(ptr noundef byval(%struct.SOverAligned) align 32 %{{.+}})

// A half occupies one SSE eightbyte.
_Float16 sse_half(_Float16 h) { return h; }

// CIR: cir.func {{.*}}@sse_half(%arg0: !cir.f16 {{.*}}) -> !cir.f16
// LLVM: define dso_local half @sse_half(half noundef %{{.+}})

// So does a bfloat.
__bf16 sse_bfloat(__bf16 b) { return b; }

// CIR: cir.func {{.*}}@sse_bfloat(%arg0: !cir.bf16 {{.*}}) -> !cir.bf16
// LLVM: define dso_local bfloat @sse_bfloat(bfloat noundef %{{.+}})

// __float128 spans an SSE/SSEUP pair, which is still one register pair.
__float128 sse_quad(__float128 q) { return q; }

// CIR: cir.func {{.*}}@sse_quad(%arg0: !cir.f128 {{.*}}) -> !cir.f128
// LLVM: define dso_local fp128 @sse_quad(fp128 noundef %{{.+}})

// x87 long double is the X87/X87UP pair, returned in st0.
long double x87_long_double(long double l) { return l; }

// CIR: cir.func {{.*}}@x87_long_double(%arg0: !cir.long_double<!cir.f80> {{.*}}) -> !cir.long_double<!cir.f80>
// LLVM: define dso_local x86_fp80 @x87_long_double(x86_fp80 noundef %{{.+}})

// Wrapping the long double in a struct merges the eightbytes to MEMORY, so
// the argument becomes byval while the return still comes back in st0.
typedef struct { long double l; } SLongDouble;
SLongDouble ret_long_double_struct(SLongDouble s) { return s; }

// CIR: cir.func {{.*}}@ret_long_double_struct(%arg0: !cir.ptr<!rec_SLongDouble> {{.*}}llvm.byval = !rec_SLongDouble{{.*}}) -> !cir.f80
// LLVM-CIR: define dso_local x86_fp80 @ret_long_double_struct(ptr noalias noundef byval(%struct.SLongDouble) align 16 %{{.+}})
// LLVM-OGCG: define dso_local x86_fp80 @ret_long_double_struct(ptr noundef byval(%struct.SLongDouble) align 16 %{{.+}})

// A union holding a long double is accepted because the long double spans the
// union's declared size.
typedef union { long double l; int i; } ULongDouble;
void take_union_long_double(ULongDouble u) { (void)u; }

// CIR: cir.func {{.*}}@take_union_long_double(%arg0: !cir.ptr<!rec_ULongDouble> {{.*}}llvm.byval = !rec_ULongDouble{{.*}})
// LLVM-CIR: define dso_local void @take_union_long_double(ptr noalias noundef byval(%union.ULongDouble) align 16 %{{.+}})
// LLVM-OGCG: define dso_local void @take_union_long_double(ptr noundef byval(%union.ULongDouble) align 16 %{{.+}})

// A _Complex of quads exceeds two eightbytes and goes to memory both ways, so
// the sret and byval pointees here are a _Complex rather than a record.
_Complex __float128 complex_quad(_Complex __float128 z) { return z; }

// CIR: cir.func {{.*}}@complex_quad(%arg0: !cir.ptr<!cir.complex<!cir.f128>> {{.*}}llvm.sret = !cir.complex<!cir.f128>{{.*}}, %arg1: !cir.ptr<!cir.complex<!cir.f128>> {{.*}}llvm.byval = !cir.complex<!cir.f128>{{.*}})
// LLVM-CIR: define dso_local void @complex_quad(ptr dead_on_unwind noalias writable sret({ fp128, fp128 }) align 16 %{{[^,)]+}}, ptr noalias noundef byval({ fp128, fp128 }) align 16 %{{[^,)]+}})
// LLVM-OGCG: define dso_local void @complex_quad(ptr dead_on_unwind noalias writable sret({ fp128, fp128 }) align 16 %{{[^,)]+}}, ptr noundef byval({ fp128, fp128 }) align 16 %{{[^,)]+}})

// Both halves of a _Complex float share one SSE eightbyte, so it coerces to
// the two-element vector that eightbyte holds.
_Complex float complex_float(_Complex float c) { return c; }

// CIR: cir.func {{.*}}@complex_float(%arg0: !cir.vector<2 x !cir.float> {{.*}}) -> !cir.vector<2 x !cir.float>
// LLVM: define dso_local <2 x float> @complex_float(<2 x float> noundef %{{.+}})

// _Complex double needs two SSE eightbytes, so it flattens into a pair.
// Flattening drops the parameter's noundef, which classic keeps on each half.
_Complex double complex_double(_Complex double c) { return c; }

// CIR: cir.func {{.*}}@complex_double(%arg0: !cir.double{{.*}}, %arg1: !cir.double{{.*}}) -> ![[F64PAIR]]
// LLVM-CIR: define dso_local { double, double } @complex_double(double %{{[^,)]+}}, double %{{[^,)]+}})
// LLVM-OGCG: define dso_local { double, double } @complex_double(double noundef %{{[^,)]+}}, double noundef %{{[^,)]+}})

// A _Complex of integers packs both halves into one INTEGER eightbyte.
_Complex int complex_int(_Complex int c) { return c; }

// CIR: cir.func {{.*}}@complex_int(%arg0: !u64i {{.*}}) -> !u64i
// LLVM: define dso_local i64 @complex_int(i64 noundef %{{.+}})

// COMPLEX_X87 passes in memory and returns as the st0/st1 pair.
_Complex long double complex_long_double(_Complex long double c) { return c; }

// CIR: cir.func {{.*}}@complex_long_double(%arg0: !cir.ptr<!cir.complex<!cir.long_double<!cir.f80>>> {{.*}}llvm.byval = !cir.complex<!cir.long_double<!cir.f80>>{{.*}}) -> ![[X87PAIR]]
// LLVM-CIR: define dso_local { x86_fp80, x86_fp80 } @complex_long_double(ptr noalias noundef byval({ x86_fp80, x86_fp80 }) align 16 %{{.+}})
// LLVM-OGCG: define dso_local { x86_fp80, x86_fp80 } @complex_long_double(ptr noundef byval({ x86_fp80, x86_fp80 }) align 16 %{{.+}})

// A _Complex of 16-bit floats fits one eightbyte, so it coerces to the
// two-element vector of that format.
_Complex _Float16 complex_half(_Complex _Float16 c) { return c; }

// CIR: cir.func {{.*}}@complex_half(%arg0: !cir.vector<2 x !cir.f16> {{.*}}) -> !cir.vector<2 x !cir.f16>
// LLVM: define dso_local <2 x half> @complex_half(<2 x half> noundef %{{[^,)]+}})

// A _Complex of 64-bit integers spans two INTEGER eightbytes, so it flattens
// into a register pair instead of coercing to one value.
_Complex long long complex_longlong(_Complex long long c) { return c; }

// CIR: cir.func {{.*}}@complex_longlong(%arg0: !u64i {{.*}}, %arg1: !u64i {{.*}}) -> ![[I64PAIR]]
// LLVM-CIR: define dso_local { i64, i64 } @complex_longlong(i64 %{{[^,)]+}}, i64 %{{[^,)]+}})
// LLVM-OGCG: define dso_local { i64, i64 } @complex_longlong(i64 noundef %{{[^,)]+}}, i64 noundef %{{[^,)]+}})

// A _Complex reaches the classifier as a record member too, not just on its
// own, so these cover the field walk rather than the top-level mapping.
typedef struct { _Complex float c; } WrapComplexFloat;
void take_wrap_complex_float(WrapComplexFloat s) { (void)s; }

// CIR: cir.func {{.*}}@take_wrap_complex_float(%arg0: !cir.vector<2 x !cir.float>{{.*}})
// LLVM: define dso_local void @take_wrap_complex_float(<2 x float> %{{[^,)]+}})

typedef struct { _Complex double c; } WrapComplexDouble;
void take_wrap_complex_double(WrapComplexDouble s) { (void)s; }

// CIR: cir.func {{.*}}@take_wrap_complex_double(%arg0: !cir.double{{.*}}, %arg1: !cir.double{{.*}})
// LLVM: define dso_local void @take_wrap_complex_double(double %{{[^,)]+}}, double %{{[^,)]+}})

// An all-float aggregate's SSE eightbyte coerces to a vector.
typedef struct { float x, y; } TwoFloats;
TwoFloats two_floats(TwoFloats s) { return s; }

// CIR: cir.func {{.*}}@two_floats(%arg0: !cir.vector<2 x !cir.float> {{.*}}) -> !cir.vector<2 x !cir.float>
// LLVM: define dso_local <2 x float> @two_floats(<2 x float> %{{[^,)]+}})

// The same holds for an array of floats inside a struct.
typedef struct { float a[2]; } FloatArray;
void take_float_array(FloatArray s) { (void)s; }

// CIR: cir.func {{.*}}@take_float_array(%arg0: !cir.vector<2 x !cir.float>{{.*}})
// LLVM: define dso_local void @take_float_array(<2 x float> %{{[^,)]+}})

// A 16-bit float pair coerces to a vector of that same format.
typedef struct { _Float16 a, b; } TwoHalves;
void take_two_halves(TwoHalves s) { (void)s; }

// CIR: cir.func {{.*}}@take_two_halves(%arg0: !cir.vector<2 x !cir.f16>{{.*}})
// LLVM: define dso_local void @take_two_halves(<2 x half> %{{[^,)]+}})

typedef struct { __bf16 a, b; } TwoBFloats;
void take_two_bfloats(TwoBFloats s) { (void)s; }

// CIR: cir.func {{.*}}@take_two_bfloats(%arg0: !cir.vector<2 x !cir.bf16>{{.*}})
// LLVM: define dso_local void @take_two_bfloats(<2 x bfloat> %{{[^,)]+}})

// A 16-bit float sharing its eightbyte with a wider float widens the vector to
// the eightbyte rather than to the members.  The element format is always
// IEEE half here, so a bfloat pairing this way comes back as half too.
typedef struct { _Float16 h; float f; } HalfThenFloat;
void take_half_then_float(HalfThenFloat s) { (void)s; }

// CIR: cir.func {{.*}}@take_half_then_float(%arg0: !cir.vector<4 x !cir.f16>{{.*}})
// LLVM: define dso_local void @take_half_then_float(<4 x half> %{{[^,)]+}})

typedef struct { __bf16 b; float f; } BFloatThenFloat;
void take_bfloat_then_float(BFloatThenFloat s) { (void)s; }

// CIR: cir.func {{.*}}@take_bfloat_then_float(%arg0: !cir.vector<4 x !cir.f16>{{.*}})
// LLVM: define dso_local void @take_bfloat_then_float(<4 x half> %{{[^,)]+}})

// An IEEE quad reaches a register, where an x87 long double of the same width
// would go to memory.
typedef struct { __float128 q; } WrapQuad;
void take_wrap_quad(WrapQuad s) { (void)s; }

// CIR: cir.func {{.*}}@take_wrap_quad(%arg0: !cir.f128{{.*}})
// LLVM: define dso_local void @take_wrap_quad(fp128 %{{[^,)]+}})

// Three floats span two eightbytes: a vector for the first pair, a scalar for
// the remainder.
typedef struct { float x, y, z; } ThreeFloats;
void take_three_floats(ThreeFloats s) { (void)s; }

// CIR: cir.func {{.*}}@take_three_floats(%arg0: !cir.vector<2 x !cir.float>{{.*}}, %arg1: !cir.float{{.*}})
// LLVM: define dso_local void @take_three_floats(<2 x float> %{{[^,)]+}}, float %{{[^,)]+}})

// Four floats fill both eightbytes, so the coercion is the one record whose
// every field is a vector, in argument and in return position.
typedef struct { float a, b, c, d; } FourFloats;
FourFloats four_floats(FourFloats s) { return s; }

// CIR: cir.func {{.*}}@four_floats(%arg0: !cir.vector<2 x !cir.float>{{.*}}, %arg1: !cir.vector<2 x !cir.float>{{.*}}) -> ![[F32X2PAIR]]
// LLVM: define dso_local { <2 x float>, <2 x float> } @four_floats(<2 x float> %{{[^,)]+}}, <2 x float> %{{[^,)]+}})

void call_complex_float(_Complex float c) { complex_float(c); }

// CIR: cir.func {{.*}}@call_complex_float(%arg0: !cir.vector<2 x !cir.float>
// CIR:   cir.call @complex_float(%{{.+}}) : (!cir.vector<2 x !cir.float> {llvm.noundef}) -> !cir.vector<2 x !cir.float>
// LLVM: define dso_local void @call_complex_float(<2 x float> noundef %{{.+}})
// LLVM:   call <2 x float> @complex_float(<2 x float> noundef %{{.+}})
