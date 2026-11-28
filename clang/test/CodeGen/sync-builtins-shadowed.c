// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s

// GH120083: a type-generic __sync_* call is rewritten into a call to the
// size-specific builtin; a declaration that merely shares that builtin's name
// must not become the callee.

static int __sync_lock_test_and_set_4(volatile int *, int);
static short __sync_fetch_and_sub_2(volatile short *, short, ...);
static void __sync_lock_release_1(volatile char *);
static void __sync_fetch_and_or_16(volatile __int128 *, __int128);
static int __sync_swap_4;
typedef long long __sync_bool_compare_and_swap_8;
enum { __sync_val_compare_and_swap_8 };

char c;
short s;
int i;
long long ll;
__int128 i128;

// CHECK-LABEL: define {{.*}}i32 @lock_test_and_set(
// CHECK-NOT:   call
// CHECK:       atomicrmw xchg ptr @i, i32 %{{.*}} seq_cst, align 4
// CHECK-NOT:   call
// CHECK:       ret i32
int lock_test_and_set(int a) { return __sync_lock_test_and_set(&i, a); }

// CHECK-LABEL: define {{.*}}i16 @fetch_and_sub(
// CHECK-NOT:   call
// CHECK:       atomicrmw sub ptr @s, i16 %{{.*}} seq_cst, align 2
// CHECK-NOT:   call
// CHECK:       ret i16
short fetch_and_sub(short a) {
  return __sync_fetch_and_sub((volatile short *)&s, a);
}

// CHECK-LABEL: define {{.*}}void @lock_release(
// CHECK-NOT:   call
// CHECK:       store atomic i8 0, ptr @c release, align 1
// CHECK-NOT:   call
// CHECK:       ret void
void lock_release(void) { __sync_lock_release(&c); }

// CHECK-LABEL: define {{.*}}void @fetch_and_or(
// CHECK-NOT:   call
// CHECK:       atomicrmw or ptr @i128, i128 %{{.*}} seq_cst, align 16
// CHECK-NOT:   call
// CHECK:       ret void
void fetch_and_or(__int128 a) { (void)__sync_fetch_and_or(&i128, a); }

// CHECK-LABEL: define {{.*}}i32 @swap(
// CHECK-NOT:   call
// CHECK:       atomicrmw xchg ptr @i, i32 %{{.*}} seq_cst, align 4
// CHECK-NOT:   call
// CHECK:       ret i32
int swap(int a) { return __sync_swap(&i, a); }

// CHECK-LABEL: define {{.*}}i32 @bool_compare_and_swap(
// CHECK-NOT:   call
// CHECK:       cmpxchg ptr @ll, i64 %{{.*}}, i64 %{{.*}} seq_cst seq_cst, align 8
// CHECK-NOT:   call
// CHECK:       ret i32
int bool_compare_and_swap(long long a, long long b) {
  return __sync_bool_compare_and_swap(&ll, a, b);
}

// CHECK-LABEL: define {{.*}}i64 @val_compare_and_swap(
// CHECK-NOT:   call
// CHECK:       cmpxchg ptr @ll, i64 %{{.*}}, i64 %{{.*}} seq_cst seq_cst, align 8
// CHECK-NOT:   call
// CHECK:       ret i64
long long val_compare_and_swap(long long a, long long b) {
  return __sync_val_compare_and_swap(&ll, a, b);
}

// In C the name is also found in enclosing block scopes.
// CHECK-LABEL: define {{.*}}i64 @local_name(
// CHECK-NOT:   call
// CHECK:       atomicrmw add ptr @ll, i64 %{{.*}} seq_cst, align 8
// CHECK-NOT:   call
// CHECK:       ret i64
long long local_name(long long a) {
  long long __sync_fetch_and_add_8 = a;
  return __sync_fetch_and_add(&ll, __sync_fetch_and_add_8);
}

// CHECK-LABEL: define {{.*}}i32 @and_and_fetch(
// CHECK-NOT:   call
// CHECK:       atomicrmw and ptr @i, i32 %{{.*}} seq_cst, align 4
// CHECK-NOT:   call
// CHECK:       ret i32
int and_and_fetch(int a) { return __sync_and_and_fetch(&i, a); }

// CHECK-NOT: @__sync_
