// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s
//
// GH111295: _Atomic(T) is never less aligned than T.

typedef __attribute__((aligned(8))) int alignedint;
typedef __attribute__((aligned(16))) char alignedchar;
enum __attribute__((aligned(8))) alignedenum { AE0 };
struct S3 { char c[3]; };
typedef struct S3 __attribute__((aligned(16))) alignedS3;

// CHECK: @a = {{(dso_local )?}}global i32 0, align 8
_Atomic(alignedint) a;
// CHECK: @c = {{(dso_local )?}}global i8 0, align 16
_Atomic(alignedchar) c;
// CHECK: @e = {{(dso_local )?}}global i32 0, align 8
_Atomic(enum alignedenum) e;
// CHECK: @s = {{(dso_local )?}}global { %struct.S3, [1 x i8] } zeroinitializer, align 16
_Atomic(alignedS3) s;
// CHECK: @plain = {{(dso_local )?}}global i32 0, align 4
_Atomic(int) plain;

// CHECK-LABEL: define {{.*}} i32 @cmp(
// CHECK: %r0.addr = alloca i32, align 8
// CHECK: load atomic i32, ptr @a seq_cst, align 8
// CHECK: icmp eq i32
int cmp(alignedint r0) { return r0 == a; }

// CHECK-LABEL: define {{.*}} void @store(
// CHECK: store atomic i32 {{.*}}, ptr @a seq_cst, align 8
void store(alignedint v) { a = v; }

// CHECK-LABEL: define {{.*}} void @init(
// CHECK: %l = alloca i32, align 8
// CHECK: store i32 {{.*}}, ptr %l, align 8
void init(alignedint v) { _Atomic(alignedint) l = v; }

// CHECK-LABEL: define {{.*}} void @c11init(
// CHECK: store i32 {{.*}}, ptr @a, align 8
void c11init(alignedint v) { __c11_atomic_init(&a, v); }

// CHECK-LABEL: define {{.*}} i32 @c11load(
// CHECK: load atomic i32, ptr @a seq_cst, align 8
alignedint c11load(void) { return __c11_atomic_load(&a, __ATOMIC_SEQ_CST); }

// CHECK-LABEL: define {{.*}} void @c11store(
// CHECK: store atomic i32 {{.*}}, ptr @a seq_cst, align 8
void c11store(alignedint v) { __c11_atomic_store(&a, v, __ATOMIC_SEQ_CST); }

// CHECK-LABEL: define {{.*}} i32 @c11xchg(
// CHECK: atomicrmw xchg ptr @a, i32 {{.*}} seq_cst, align 8
alignedint c11xchg(alignedint v) {
  return __c11_atomic_exchange(&a, v, __ATOMIC_SEQ_CST);
}

// CHECK-LABEL: define {{.*}} i32 @load_ptr(
// CHECK: load atomic i32, ptr %{{.*}} seq_cst, align 8
alignedint load_ptr(_Atomic(alignedint) *p) { return *p; }

// CHECK-LABEL: define {{.*}} i8 @load_char(
// CHECK: load atomic i8, ptr @c seq_cst, align 16
alignedchar load_char(void) { return c; }

// CHECK-LABEL: define {{.*}} i32 @load_enum(
// CHECK: load atomic i32, ptr @e seq_cst, align 8
enum alignedenum load_enum(void) { return e; }

// CHECK-LABEL: define {{.*}}@load_struct(
// CHECK: load atomic i32, ptr @s seq_cst, align 16
alignedS3 load_struct(void) { return s; }

// CHECK-LABEL: define {{.*}} i32 @load_plain(
// CHECK: load atomic i32, ptr @plain seq_cst, align 4
int load_plain(void) { return plain; }
