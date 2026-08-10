// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -clangir-enable-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -clangir-enable-call-conv-lowering -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-CIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-OGCG --input-file=%t.ll %s

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
// LLVM-CIR:   alloca %union.UBigOverAligned, i64 1, align 32
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
