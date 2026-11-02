// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx -fclangir -emit-llvm %s -o %t-cir-avx.ll
// RUN: FileCheck --check-prefix=AVX --input-file=%t-cir-avx.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx -emit-llvm %s -o %t-avx.ll
// RUN: FileCheck --check-prefix=AVX --input-file=%t-avx.ll %s

typedef _Bool b4 __attribute__((ext_vector_type(4)));
typedef _Bool b8 __attribute__((ext_vector_type(8)));
typedef _Bool b16 __attribute__((ext_vector_type(16)));
typedef _Bool b17 __attribute__((ext_vector_type(17)));
typedef _Bool b24 __attribute__((ext_vector_type(24)));
typedef _Bool b40 __attribute__((ext_vector_type(40)));
typedef _Bool b64 __attribute__((ext_vector_type(64)));
typedef _Bool b100 __attribute__((ext_vector_type(100)));
typedef _Bool b128 __attribute__((ext_vector_type(128)));
typedef _Bool b130 __attribute__((ext_vector_type(130)));
typedef _Bool b256 __attribute__((ext_vector_type(256)));

// A b4 member takes one byte.
struct SB4 { b4 v; };
struct SB4x2 { b4 a; b4 b; int i; };
struct SB4Arr { b4 a[3]; };
struct SB4F { b4 v; float f; };
struct SB64x2 { b64 a; b64 b; };
struct SB64x3 { b64 a, b, c; };
struct SB128 { b128 v; };
union UB4 { b4 v; char c; };

// CIR-DAG: !rec_SB4 = !cir.struct<"SB4" {data !cir.vector<4 x !cir.bool>}>
// CIR-DAG: !rec_anon_struct = !cir.struct<{data !cir.double, data !cir.double}>
// CIR-DAG: !rec_SB4x2 = !cir.struct<"SB4x2" {data !cir.vector<4 x !cir.bool>, data !cir.vector<4 x !cir.bool>, data !s32i}>
// CIR-DAG: !rec_SB4F = !cir.struct<"SB4F" {data !cir.vector<4 x !cir.bool>, data !cir.float}>
// CIR-DAG: !rec_SB64x3 = !cir.struct<"SB64x3" {data !cir.vector<64 x !cir.bool>, data !cir.vector<64 x !cir.bool>, data !cir.vector<64 x !cir.bool>}>
// CIR-DAG: !rec_SB128 = !cir.struct<"SB128" {data !cir.vector<128 x !cir.bool>}>
// LLVM-DAG: %struct.SB4 = type { i8 }
// LLVM-DAG: %struct.SB4x2 = type { i8, i8, i32 }
// LLVM-DAG: %struct.SB4Arr = type { [3 x i8] }
// LLVM-DAG: %struct.SB4F = type { i8, float }
// LLVM-DAG: %struct.SB64x2 = type { i64, i64 }
// LLVM-DAG: %struct.SB64x3 = type { i64, i64, i64 }
// LLVM-DAG: %struct.SB128 = type { i128 }
// LLVM-DAG: %union.UB4 = type { i8 }

b4 vb4(b4 v) { return v; }

// CIR: cir.func {{.*}} @vb4(%arg0: !u8i {llvm.noundef} loc({{[^)]+}})) -> !u8i
// CIR: %[[SLOT:[^ ]+]] = cir.alloca "coerce" align(1) : !cir.ptr<!u8i>
// CIR: cir.store %arg0, %[[SLOT]] : !u8i, !cir.ptr<!u8i>
// CIR: %[[VIEW:[^ ]+]] = cir.cast bitcast %[[SLOT]] : !cir.ptr<!u8i> -> !cir.ptr<!cir.vector<4 x !cir.bool>>
// CIR: cir.load %[[VIEW]] : !cir.ptr<!cir.vector<4 x !cir.bool>>, !cir.vector<4 x !cir.bool>

// LLVM: define dso_local i8 @vb4(i8 noundef %[[ARG:[^,)]+]])
// LLVM: store i8 %[[ARG]], ptr %[[SLOT:[^,]+]], align 1
// LLVM: %[[BITS:[^ ]+]] = load i8, ptr %[[SLOT]], align 1
// LLVM: bitcast i8 %[[BITS]] to <8 x i1>
// LLVM: ret i8

b8 vb8(b8 v) { return v; }

// CIR: cir.func {{.*}} @vb8(%arg0: !u8i {llvm.noundef} loc({{[^)]+}})) -> !u8i
// LLVM: define dso_local i8 @vb8(i8 noundef %{{[^,)]+}})

b16 vb16(b16 v) { return v; }

// CIR: cir.func {{.*}} @vb16(%arg0: !u16i {llvm.noundef} loc({{[^)]+}})) -> !u16i
// LLVM: define dso_local i16 @vb16(i16 noundef %{{[^,)]+}})

// The 17-bit storage integer is not a whole number of bytes, so the argument
// is not noundef.
b17 vb17(b17 v) { return v; }

// CIR: cir.func {{.*}} @vb17(%arg0: !u32i loc({{[^)]+}})) -> !u32i
// CIR: %[[SLOT17:[^ ]+]] = cir.alloca "coerce" align(4) : !cir.ptr<!u32i>
// CIR: cir.store %arg0, %[[SLOT17]] : !u32i, !cir.ptr<!u32i>
// CIR: cir.cast bitcast %[[SLOT17]] : !cir.ptr<!u32i> -> !cir.ptr<!cir.vector<17 x !cir.bool>>

// LLVM: define dso_local i32 @vb17(i32 %[[ARG17:[^,)]+]])
// LLVM: store i32 %[[ARG17]], ptr %[[SLOT17:[^,]+]], align 4
// LLVM: load i17, ptr %[[SLOT17]], align 4
// LLVM: ret i32

// The coerced types are wider than the 24 and 40 bits of storage, so neither
// is noundef.
b24 vb24(b24 v) { return v; }
b40 vb40(b40 v) { return v; }

// CIR: cir.func {{.*}} @vb24(%arg0: !u32i loc({{[^)]+}})) -> !u32i
// CIR: cir.func {{.*}} @vb40(%arg0: !cir.double loc({{[^)]+}})) -> !cir.double
// LLVM: define dso_local i32 @vb24(i32 %{{[^,)]+}})
// LLVM: define dso_local double @vb40(double %{{[^,)]+}})

b24 call_vb24(b24 v) { return vb24(v); }

// CIR: cir.func {{.*}} @call_vb24(%arg0: !u32i loc({{[^)]+}})) -> !u32i
// CIR: cir.call @vb24(%{{[^)]+}}) : (!u32i) -> !u32i
// LLVM: define dso_local i32 @call_vb24(i32 %{{[^,)]+}})
// LLVM: call i32 @vb24(i32 %{{[^,)]+}})

b64 vb64(b64 v) { return v; }

// CIR: cir.func {{.*}} @vb64(%arg0: !cir.double {llvm.noundef} loc({{[^)]+}})) -> !cir.double
// LLVM: define dso_local double @vb64(double noundef %{{[^,)]+}})

// A 100-bit vector is passed as itself, but its storage is not a whole number
// of bytes, so it is not noundef.
b100 vb100(b100 v) { return v; }

// CIR: cir.func {{.*}} @vb100(%arg0: !cir.vector<100 x !cir.bool> loc({{[^)]+}})) -> !cir.vector<100 x !cir.bool>
// LLVM: define dso_local <100 x i1> @vb100(<100 x i1> %{{[^,)]+}})

b128 vb128(b128 v) { return v; }

// CIR: cir.func {{.*}} @vb128(%arg0: !cir.vector<128 x !cir.bool> {llvm.noundef} loc({{[^)]+}})) -> !cir.vector<128 x !cir.bool>
// CIR-NOT: cir.alloca "coerce"
// CIR: cir.return %{{.+}} : !cir.vector<128 x !cir.bool>

// LLVM: define dso_local <128 x i1> @vb128(<128 x i1> noundef %[[ARG128:[^,)]+]])
// LLVM: bitcast <128 x i1> %[[ARG128]] to i128
// LLVM: ret <128 x i1>

// Without AVX a 256-bit vector is passed in memory.
void vb256(b256 v) {}

// CIR: cir.func {{.*}} @vb256(%arg0: !cir.ptr<!cir.vector<256 x !cir.bool>> {llvm.align = 32 : i64, llvm.byval = !cir.vector<256 x !cir.bool>, llvm.noundef} loc({{[^)]+}}))
// LLVM: define dso_local void @vb256(ptr noundef byval(i256) align 32 %{{[^,)]+}})
// AVX: define dso_local void @vb256(<256 x i1> noundef %{{[^,)]+}})

struct SB4 sb4(struct SB4 s) { return s; }

// CIR: cir.func {{.*}} @sb4(%arg0: !u8i loc({{[^)]+}})) -> !u8i
// LLVM: define dso_local i8 @sb4(i8 %{{[^,)]+}})

struct SB4x2 sb4x2(struct SB4x2 s) { return s; }

// CIR: cir.func {{.*}} @sb4x2(%arg0: !u64i loc({{[^)]+}})) -> !u64i
// LLVM: define dso_local i64 @sb4x2(i64 %{{[^,)]+}})

struct SB4Arr sb4arr(struct SB4Arr s) { return s; }

// CIR: cir.func {{.*}} @sb4arr(%arg0: !cir.int<u, 24> loc({{[^)]+}})) -> !cir.int<u, 24>
// LLVM: define dso_local i24 @sb4arr(i24 %{{[^,)]+}})

struct SB64x2 sb64x2(struct SB64x2 s) { return s; }

// CIR: cir.func {{.*}} @sb64x2(%arg0: !cir.double loc({{[^)]+}}), %arg1: !cir.double loc({{[^)]+}})) -> !rec_anon_struct
// LLVM: define dso_local { double, double } @sb64x2(double %{{[^,)]+}}, double %{{[^,)]+}})

union UB4 ub4(union UB4 u) { return u; }

// CIR: cir.func {{.*}} @ub4(%arg0: !s8i loc({{[^)]+}})) -> !s8i
// LLVM: define dso_local i8 @ub4(i8 %{{[^,)]+}})

// Past the integer registers a bool vector still coerces to an integer.
void stack_vb4(long a, long b, long c, long d, long e, long f, b4 v) {}

// CIR: cir.func {{.*}} @stack_vb4(%arg0: !s64i {llvm.noundef} loc({{[^)]+}}), %arg1: !s64i {llvm.noundef} loc({{[^)]+}}), %arg2: !s64i {llvm.noundef} loc({{[^)]+}}), %arg3: !s64i {llvm.noundef} loc({{[^)]+}}), %arg4: !s64i {llvm.noundef} loc({{[^)]+}}), %arg5: !s64i {llvm.noundef} loc({{[^)]+}}), %arg6: !u8i {llvm.noundef} loc({{[^)]+}}))
// LLVM: define dso_local void @stack_vb4(i64 noundef %{{[^,)]+}}, i64 noundef %{{[^,)]+}}, i64 noundef %{{[^,)]+}}, i64 noundef %{{[^,)]+}}, i64 noundef %{{[^,)]+}}, i64 noundef %{{[^,)]+}}, i8 noundef %{{[^,)]+}})

b4 call_vb4(b4 v) { return vb4(v); }

// CIR: cir.func {{.*}} @call_vb4(%arg0: !u8i {llvm.noundef} loc({{[^)]+}})) -> !u8i
// CIR: cir.call @vb4(%{{[^)]+}}) : (!u8i {llvm.noundef}) -> !u8i

// LLVM: define dso_local i8 @call_vb4(i8 noundef %{{[^,)]+}})
// LLVM: call i8 @vb4(i8 noundef %{{[^,)]+}})

void var(int n, ...);
void call_var(b4 v) { var(1, v); }

// CIR: cir.func {{.*}} @call_var(%arg0: !u8i {llvm.noundef} loc({{[^)]+}}))
// CIR: cir.call @var(%{{[^,]+}}, %{{[^)]+}}) : (!s32i {llvm.noundef}, !u8i {llvm.noundef}) -> ()
// LLVM: call void (i32, ...) @var(i32 noundef 1, i8 noundef %{{[^,)]+}})

void call_var24(b24 v) { var(1, v); }

// CIR: cir.call @var(%{{[^,]+}}, %{{[^)]+}}) : (!s32i {llvm.noundef}, !u32i) -> ()
// LLVM: call void (i32, ...) @var(i32 noundef 1, i32 %{{[^,)]+}})

b128 call_vb128(b128 v) { return vb128(v); }

// CIR: cir.call @vb128(%{{[^)]+}}) : (!cir.vector<128 x !cir.bool> {llvm.noundef}) -> !cir.vector<128 x !cir.bool>
// LLVM: call <128 x i1> @vb128(<128 x i1> noundef %{{[^,)]+}})

b40 call_vb40(b40 v) { return vb40(v); }

// CIR: cir.call @vb40(%{{[^)]+}}) : (!cir.double) -> !cir.double
// LLVM: call double @vb40(double %{{[^,)]+}})

void call_vb256(b256 v) { vb256(v); }

// CIR: cir.call @vb256(%{{[^)]+}}) : (!cir.ptr<!cir.vector<256 x !cir.bool>> {llvm.align = 32 : i64, llvm.byval = !cir.vector<256 x !cir.bool>, llvm.noundef}) -> ()
// LLVM: call void @vb256(ptr noundef byval(i256) align 32 %{{[^,)]+}})
// AVX: call void @vb256(<256 x i1> noundef %{{[^,)]+}})

// The byval pointer is noundef even though the 130-bit storage is not a whole
// number of bytes.
void vb130(b130 v) {}

// CIR: cir.func {{.*}} @vb130(%arg0: !cir.ptr<!cir.vector<130 x !cir.bool>> {llvm.align = 32 : i64, llvm.byval = !cir.vector<130 x !cir.bool>, llvm.noundef} loc({{[^)]+}}))
// LLVM: define dso_local void @vb130(ptr noundef byval(i130) align 32 %{{[^,)]+}})

// A record of one 128-bit bool vector is coerced to the vector.
struct SB128 sb128(struct SB128 s) { return s; }

// CIR: cir.func {{.*}} @sb128(%arg0: !cir.vector<128 x !cir.bool> loc({{[^)]+}})) -> !cir.vector<128 x !cir.bool>
// LLVM: define dso_local <128 x i1> @sb128(<128 x i1> %{{[^,)]+}})

struct SB4F sb4f(struct SB4F s) { return s; }

// CIR: cir.func {{.*}} @sb4f(%arg0: !u64i loc({{[^)]+}})) -> !u64i
// LLVM: define dso_local i64 @sb4f(i64 %{{[^,)]+}})

struct SB64x3 sb64x3(struct SB64x3 s) { return s; }

// CIR: cir.func {{.*}} @sb64x3(%arg0: !cir.ptr<!rec_SB64x3> {llvm.align = 8 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_SB64x3, llvm.writable} loc({{[^)]+}}), %arg1: !cir.ptr<!rec_SB64x3> {llvm.align = 8 : i64, llvm.byval = !rec_SB64x3, llvm.noundef} loc({{[^)]+}}))
// LLVM: define dso_local void @sb64x3(ptr dead_on_unwind noalias writable sret(%struct.SB64x3) align 8 %{{[^,)]+}}, ptr noundef byval(%struct.SB64x3) align 8 %{{[^,)]+}})

// A call returning through an sret slot drops noundef from a coerced
// argument too.
struct Big { long a, b, c; };
struct Big mk24(b24 v);
struct Big sret_vb24(b24 v) { return mk24(v); }

// CIR: cir.func {{.*}} @sret_vb24(%arg0: !cir.ptr<!rec_Big> {llvm.align = 8 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_Big, llvm.writable} loc({{[^)]+}}), %arg1: !u32i loc({{[^)]+}}))
// CIR: cir.call @mk24(%arg0, %{{[^)]+}}) : (!cir.ptr<!rec_Big> {llvm.align = 8 : i64, llvm.dead_on_unwind, llvm.sret = !rec_Big, llvm.writable}, !u32i) -> ()
// LLVM: define dso_local void @sret_vb24(ptr dead_on_unwind noalias writable sret(%struct.Big) align 8 %{{[^,)]+}}, i32 %{{[^,)]+}})
// LLVM: call void @mk24(ptr dead_on_unwind writable sret(%struct.Big) align 8 %{{[^,)]+}}, i32 %{{[^,)]+}})

b4 get_va4(int n, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, n);
  b4 v = __builtin_va_arg(ap, b4);
  __builtin_va_end(ap);
  return v;
}

// CIR: cir.func {{.*}} @get_va4(%arg0: !s32i {llvm.noundef} loc({{[^)]+}}), ...) -> !u8i
// LLVM: define dso_local i8 @get_va4(i32 noundef %{{[^,)]+}}, ...)
// LLVM: icmp ule i32 %{{[^,]+}}, 40
// LLVM: add i32 %{{[^,]+}}, 8
// LLVM: load i8, ptr %{{[^,]+}}, align 1
