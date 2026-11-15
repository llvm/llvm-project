// RUN: %clang_cc1 -triple arm64e-apple-ios -fptrauth-calls -fptrauth-intrinsics -emit-llvm %s -o - | FileCheck %s

typedef int *__ptrauth(2, 0, 50) UniformPtr;
typedef int *__ptrauth(2, 1, 100) AuthPtr;
typedef long __ptrauth(2, 0, 200) UniformInt;
typedef void Fn(void);
typedef Fn *__ptrauth(0, 0, 300) UniformFn;

_Atomic UniformPtr atomic_ptr, atomic_src;
_Atomic AuthPtr atomic_auth, atomic_auth_src;
UniformPtr qualified_ptr;
_Atomic UniformInt atomic_int;
UniformInt qualified_int;
_Atomic(Fn *) plain_fn;
_Atomic UniformFn atomic_fn;

// CHECK-LABEL: define {{.*}}ptr @load_implicit()
// CHECK:         [[SIGNED:%.*]] = load atomic ptr, ptr @atomic_ptr
// CHECK:         [[INT:%.*]] = ptrtoint ptr [[SIGNED]] to i64
// CHECK-NEXT:    call i64 @llvm.ptrauth.auth(i64 [[INT]], i32 2, i64 50)
int *load_implicit(void) { return atomic_ptr; }

// CHECK-LABEL: define {{.*}}void @store_implicit(
// CHECK:         call i64 @llvm.ptrauth.sign(i64 {{.*}}, i32 2, i64 50)
// CHECK:         [[SIGNED:%.*]] = phi ptr
// CHECK-NEXT:    store atomic ptr [[SIGNED]], ptr @atomic_ptr
void store_implicit(int *p) { atomic_ptr = p; }

// FIXME: authenticates and re-signs under the same schema.
// CHECK-LABEL: define {{.*}}void @copy_implicit()
// CHECK:         load atomic ptr, ptr @atomic_src
// CHECK:         call i64 @llvm.ptrauth.auth(i64 {{.*}}, i32 2, i64 50)
// CHECK:         call i64 @llvm.ptrauth.sign(i64 {{.*}}, i32 2, i64 50)
// CHECK:         [[SIGNED:%.*]] = phi ptr
// CHECK-NEXT:    store atomic ptr [[SIGNED]], ptr @atomic_ptr
void copy_implicit(void) { atomic_ptr = atomic_src; }

// CHECK-LABEL: define {{.*}}ptr @load_implicit_auth()
// CHECK:         [[SIGNED:%.*]] = load atomic ptr, ptr @atomic_auth
// CHECK-NEXT:    [[DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @atomic_auth to i64), i64 100)
// CHECK:         [[INT:%.*]] = ptrtoint ptr [[SIGNED]] to i64
// CHECK-NEXT:    call i64 @llvm.ptrauth.auth(i64 [[INT]], i32 2, i64 [[DISC]])
int *load_implicit_auth(void) { return atomic_auth; }

// CHECK-LABEL: define {{.*}}void @store_implicit_auth(
// CHECK:         [[DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @atomic_auth to i64), i64 100)
// CHECK:         call i64 @llvm.ptrauth.sign(i64 {{.*}}, i32 2, i64 [[DISC]])
// CHECK:         [[SIGNED:%.*]] = phi ptr
// CHECK-NEXT:    store atomic ptr [[SIGNED]], ptr @atomic_auth
void store_implicit_auth(int *p) { atomic_auth = p; }

// FIXME: authenticates and re-signs instead of one resign.
// CHECK-LABEL: define {{.*}}void @copy_implicit_auth()
// CHECK:         [[SRC:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @atomic_auth_src to i64), i64 100)
// CHECK:         call i64 @llvm.ptrauth.auth(i64 {{.*}}, i32 2, i64 [[SRC]])
// CHECK:         [[DST:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @atomic_auth to i64), i64 100)
// CHECK:         call i64 @llvm.ptrauth.sign(i64 {{.*}}, i32 2, i64 [[DST]])
// CHECK:         [[SIGNED:%.*]] = phi ptr
// CHECK-NEXT:    store atomic ptr [[SIGNED]], ptr @atomic_auth
void copy_implicit_auth(void) { atomic_auth = atomic_auth_src; }

// CHECK-LABEL: define {{.*}}void @store_fn_implicit(
// CHECK:         call i64 @llvm.ptrauth.resign(i64 {{.*}}, i32 0, i64 0, i32 0, i64 300)
// CHECK:         [[SIGNED:%.*]] = phi ptr
// CHECK-NEXT:    store atomic ptr [[SIGNED]], ptr @atomic_fn
void store_fn_implicit(Fn *p) { atomic_fn = p; }

// CHECK-LABEL: define {{.*}}ptr @load_fn_implicit()
// CHECK:         [[SIGNED:%.*]] = load atomic ptr, ptr @atomic_fn
// CHECK:         [[INT:%.*]] = ptrtoint ptr [[SIGNED]] to i64
// CHECK-NEXT:    call i64 @llvm.ptrauth.resign(i64 [[INT]], i32 0, i64 300, i32 0, i64 0)
Fn *load_fn_implicit(void) { return atomic_fn; }

// CHECK-LABEL: define {{.*}}void @init_decl(
// CHECK:         call i64 @llvm.ptrauth.sign(i64 {{.*}}, i32 2, i64 50)
// CHECK:         [[SIGNED:%.*]] = phi ptr
// CHECK-NEXT:    store ptr [[SIGNED]], ptr %l
void init_decl(int *p) { _Atomic UniformPtr l = p; }

// FIXME: stores the raw pointer.
// CHECK-LABEL: define {{.*}}void @init_c11(
// CHECK-NOT:     @llvm.ptrauth
// CHECK:         store ptr {{.*}}, ptr @atomic_ptr
void init_c11(int *p) { __c11_atomic_init(&atomic_ptr, p); }

// FIXME: returns the signed pointer.
// CHECK-LABEL: define {{.*}}ptr @load_c11()
// CHECK:         load atomic ptr, ptr @atomic_ptr
// CHECK-NOT:     @llvm.ptrauth
int *load_c11(void) { return __c11_atomic_load(&atomic_ptr, __ATOMIC_SEQ_CST); }

// FIXME: stores the raw pointer.
// CHECK-LABEL: define {{.*}}void @store_c11(
// CHECK-NOT:     @llvm.ptrauth
// CHECK:         store atomic ptr {{.*}}, ptr @atomic_ptr
void store_c11(int *p) { __c11_atomic_store(&atomic_ptr, p, __ATOMIC_SEQ_CST); }

// FIXME: stores the raw pointer and returns the signed one.
// CHECK-LABEL: define {{.*}}ptr @exchange_c11(
// CHECK-NOT:     @llvm.ptrauth
// CHECK:         atomicrmw xchg ptr @atomic_ptr, ptr {{.*}}
// CHECK-NOT:     @llvm.ptrauth
int *exchange_c11(int *p) {
  return __c11_atomic_exchange(&atomic_ptr, p, __ATOMIC_SEQ_CST);
}

// FIXME: compares the raw expected and stores the raw desired.
// CHECK-LABEL: define {{.*}}i1 @cmpxchg_c11(
// CHECK-NOT:     @llvm.ptrauth
// CHECK:         cmpxchg ptr @atomic_ptr, ptr {{.*}}, ptr {{.*}}
// CHECK-NOT:     @llvm.ptrauth
_Bool cmpxchg_c11(int **expected, int *desired) {
  return __c11_atomic_compare_exchange_strong(&atomic_ptr, expected, desired,
                                              __ATOMIC_SEQ_CST,
                                              __ATOMIC_SEQ_CST);
}

// FIXME: compares the raw expected and stores the raw desired.
// CHECK-LABEL: define {{.*}}i1 @cmpxchg_weak_c11(
// CHECK-NOT:     @llvm.ptrauth
// CHECK:         cmpxchg weak ptr @atomic_ptr, ptr {{.*}}, ptr {{.*}}
// CHECK-NOT:     @llvm.ptrauth
_Bool cmpxchg_weak_c11(int **expected, int *desired) {
  return __c11_atomic_compare_exchange_weak(&atomic_ptr, expected, desired,
                                            __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}

// FIXME: returns the signed pointer.
// CHECK-LABEL: define {{.*}}ptr @load_n()
// CHECK:         load atomic ptr, ptr @qualified_ptr
// CHECK-NOT:     @llvm.ptrauth
int *load_n(void) { return __atomic_load_n(&qualified_ptr, __ATOMIC_SEQ_CST); }

// FIXME: stores the raw pointer.
// CHECK-LABEL: define {{.*}}void @store_n(
// CHECK-NOT:     @llvm.ptrauth
// CHECK:         store atomic ptr {{.*}}, ptr @qualified_ptr
void store_n(int *p) { __atomic_store_n(&qualified_ptr, p, __ATOMIC_SEQ_CST); }

// FIXME: stores the raw pointer and returns the signed one.
// CHECK-LABEL: define {{.*}}ptr @exchange_n(
// CHECK-NOT:     @llvm.ptrauth
// CHECK:         atomicrmw xchg ptr @qualified_ptr, ptr {{.*}}
// CHECK-NOT:     @llvm.ptrauth
int *exchange_n(int *p) {
  return __atomic_exchange_n(&qualified_ptr, p, __ATOMIC_SEQ_CST);
}

// FIXME: compares the raw expected and stores the raw desired.
// CHECK-LABEL: define {{.*}}i1 @cmpxchg_n(
// CHECK-NOT:     @llvm.ptrauth
// CHECK:         cmpxchg ptr @qualified_ptr, ptr {{.*}}, ptr {{.*}}
// CHECK-NOT:     @llvm.ptrauth
_Bool cmpxchg_n(int **expected, int *desired) {
  return __atomic_compare_exchange_n(&qualified_ptr, expected, desired, 0,
                                     __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}

// FIXME: compares the raw expected.
// CHECK-LABEL: define {{.*}}i1 @cmpxchg_generic(
// CHECK-NOT:     @llvm.ptrauth
// CHECK:         cmpxchg ptr @qualified_ptr, ptr {{.*}}, ptr {{.*}}
// CHECK-NOT:     @llvm.ptrauth
_Bool cmpxchg_generic(int **expected, UniformPtr *desired) {
  return __atomic_compare_exchange(&qualified_ptr, expected, desired, 0,
                                   __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}

// FIXME: stores the signed bits into a raw pointer.
// CHECK-LABEL: define {{.*}}void @load_generic(
// CHECK:         [[SIGNED:%.*]] = load atomic ptr, ptr @qualified_ptr
// CHECK-NEXT:    store ptr [[SIGNED]], ptr
void load_generic(int **out) {
  __atomic_load(&qualified_ptr, out, __ATOMIC_SEQ_CST);
}

// CHECK-LABEL: define {{.*}}void @store_generic(
// CHECK-NOT:     @llvm.ptrauth
// CHECK:         store atomic ptr {{.*}}, ptr @qualified_ptr
void store_generic(UniformPtr *in) {
  __atomic_store(&qualified_ptr, in, __ATOMIC_SEQ_CST);
}

// FIXME: stores the raw pointer.
// CHECK-LABEL: define {{.*}}void @exchange_generic(
// CHECK-NOT:     @llvm.ptrauth
// CHECK:         atomicrmw xchg ptr @qualified_ptr, ptr {{.*}}
// CHECK-NOT:     @llvm.ptrauth
void exchange_generic(int **in, UniformPtr *out) {
  __atomic_exchange(&qualified_ptr, in, out, __ATOMIC_SEQ_CST);
}

// CHECK-LABEL: define {{.*}}i1 @cmpxchg_plain_fn(
// CHECK-NOT:     @llvm.ptrauth
// CHECK:         cmpxchg ptr @plain_fn, ptr {{.*}}, ptr {{.*}}
// CHECK-NOT:     @llvm.ptrauth
_Bool cmpxchg_plain_fn(Fn **expected, Fn *desired) {
  return __c11_atomic_compare_exchange_strong(&plain_fn, expected, desired,
                                              __ATOMIC_SEQ_CST,
                                              __ATOMIC_SEQ_CST);
}

// FIXME: compares and stores pointers signed under the function pointer schema.
// CHECK-LABEL: define {{.*}}i1 @cmpxchg_fn(
// CHECK-NOT:     @llvm.ptrauth
// CHECK:         cmpxchg ptr @atomic_fn, ptr {{.*}}, ptr {{.*}}
// CHECK-NOT:     @llvm.ptrauth
_Bool cmpxchg_fn(Fn **expected, Fn *desired) {
  return __c11_atomic_compare_exchange_strong(&atomic_fn, expected, desired,
                                              __ATOMIC_SEQ_CST,
                                              __ATOMIC_SEQ_CST);
}

// FIXME: returns the signed integer.
// CHECK-LABEL: define {{.*}}i64 @load_int()
// CHECK:         load atomic i64, ptr @atomic_int
// CHECK-NOT:     @llvm.ptrauth
long load_int(void) { return __c11_atomic_load(&atomic_int, __ATOMIC_SEQ_CST); }

// FIXME: stores the raw integer.
// CHECK-LABEL: define {{.*}}void @store_int(
// CHECK-NOT:     @llvm.ptrauth
// CHECK:         store atomic i64 {{.*}}, ptr @atomic_int
void store_int(long v) { __c11_atomic_store(&atomic_int, v, __ATOMIC_SEQ_CST); }

// FIXME: adds to the signed bits.
// CHECK-LABEL: define {{.*}}i64 @fetch_add_int()
// CHECK-NOT:     @llvm.ptrauth
// CHECK:         atomicrmw add ptr @atomic_int, i64 {{.*}}
// CHECK-NOT:     @llvm.ptrauth
long fetch_add_int(void) {
  return __c11_atomic_fetch_add(&atomic_int, 1, __ATOMIC_SEQ_CST);
}

// FIXME: adds to the signed bits.
// CHECK-LABEL: define {{.*}}void @add_assign_int()
// CHECK-NOT:     @llvm.ptrauth
// CHECK:         atomicrmw add ptr @atomic_int, i64 1
// CHECK-NOT:     @llvm.ptrauth
void add_assign_int(void) { atomic_int += 1; }

// FIXME: adds to the signed bits.
// CHECK-LABEL: define {{.*}}void @inc_int()
// CHECK-NOT:     @llvm.ptrauth
// CHECK:         atomicrmw add ptr @atomic_int, i64 1
// CHECK-NOT:     @llvm.ptrauth
void inc_int(void) { atomic_int++; }

// FIXME: adds to the signed bits.
// CHECK-LABEL: define {{.*}}i64 @sync_fetch_add_int()
// CHECK-NOT:     @llvm.ptrauth
// CHECK:         atomicrmw add ptr @qualified_int, i64 1
// CHECK-NOT:     @llvm.ptrauth
long sync_fetch_add_int(void) { return __sync_fetch_and_add(&qualified_int, 1); }

// FIXME: writes a raw byte into the signed representation.
// CHECK-LABEL: define {{.*}}i1 @test_and_set()
// CHECK:         atomicrmw xchg ptr @qualified_ptr, i8 1
_Bool test_and_set(void) {
  return __atomic_test_and_set(&qualified_ptr, __ATOMIC_SEQ_CST);
}

// FIXME: writes a raw byte into the signed representation.
// CHECK-LABEL: define {{.*}}void @clear()
// CHECK:         store atomic i8 0, ptr @qualified_ptr
void clear(void) { __atomic_clear(&qualified_ptr, __ATOMIC_SEQ_CST); }

// FIXME: authenticates once, then compares and increments the signed bits without re-signing.
// CHECK-LABEL: define {{.*}}void @inc_ptr()
// CHECK:         call i64 @llvm.ptrauth.auth(i64 {{.*}}, i32 2, i64 50)
// CHECK:         [[AUTHED:%.*]] = phi ptr [ null, {{.*}} ]
// CHECK:         [[CUR:%.*]] = phi ptr [ [[AUTHED]], {{.*}} ], [ [[OBSERVED:%.*]], {{.*}} ]
// CHECK-NEXT:    [[NEW:%.*]] = getelementptr {{.*}}, ptr [[CUR]], i32 1
// CHECK-NEXT:    [[PAIR:%.*]] = cmpxchg ptr @atomic_ptr, ptr [[CUR]], ptr [[NEW]]
// CHECK-NEXT:    [[OBSERVED]] = extractvalue { ptr, i1 } [[PAIR]], 0
// CHECK-NOT:     @llvm.ptrauth
void inc_ptr(void) { atomic_ptr++; }
