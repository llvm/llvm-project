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
