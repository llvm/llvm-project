// RUN: %clang_cc1 -triple x86_64-linux-gnu -fexperimental-overflow-behavior-types -std=c2x %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK

#define __wrap __attribute__((overflow_behavior(wrap)))
#define __no_trap __attribute__((overflow_behavior(trap)))

typedef int __ob_wrap w_int;
typedef int __ob_trap no_trap_int;


// CHECK-LABEL: define {{.*}} @generic_selection_test_nomatch
int generic_selection_test_nomatch(int x) {
  // CHECK: ret i32 3
  return _Generic(x, w_int: 1, no_trap_int: 2, default: 3);
}

// CHECK-LABEL: define {{.*}} @generic_selection_test_obtmatch
int generic_selection_test_obtmatch(w_int x) {
  // CHECK: ret i32 1
  return _Generic(x, w_int: 1, no_trap_int: 2, default: 3);
}

// CHECK-LABEL: define {{.*}} @generic_selection_test_obt_nomatch
int generic_selection_test_obt_nomatch(w_int x) {
  // CHECK: ret i32 3
  return _Generic(x, int: 1, char: 2, default: 3);
}

// CHECK-LABEL: define {{.*}} @signed_bitint_test
void signed_bitint_test(_BitInt(4) __ob_trap a, _BitInt(8) __ob_trap b, _BitInt(99) __ob_wrap c) {
  // CHECK: call { i32, i1 } @llvm.sadd.with.overflow.i32(i32
  (a + 1);

  // CHECK: call { i32, i1 } @llvm.sadd.with.overflow.i32(i32
  (b + 1);

  // CHECK: add i99 {{.*}}, 1
  (c + 1);
}

// CHECK-LABEL: define {{.*}} @unsigned_bitint_test
void unsigned_bitint_test(unsigned _BitInt(4) __ob_trap a, unsigned _BitInt(8) __ob_trap b, unsigned _BitInt(99) __ob_wrap c) {
  // CHECK: call { i32, i1 } @llvm.sadd.with.overflow.i32(i32
  (a + 1);

  // CHECK: call { i32, i1 } @llvm.sadd.with.overflow.i32(i32
  (b + 1);

   // CHECK: add i99 {{.*}}, 1
  (c + 1);
}

// CHECK-LABEL: define {{.*}} @generic_obt_vs_underlying
int generic_obt_vs_underlying(int plain, int __ob_wrap wrapped) {
  // Regular int should match int case, store 1
  // CHECK: store i32 1, ptr %plain_result
  int plain_result = _Generic(plain,
    int: 1,
    int __ob_wrap: 2,
    default: 3);

  // Wrapped int should match __ob_wrap case, store 2
  // CHECK: store i32 2, ptr %wrapped_result
  int wrapped_result = _Generic(wrapped,
    int: 1,
    int __ob_wrap: 2,
    default: 3);

  // CHECK: add nsw i32
  // CHECK: ret i32
  return plain_result + wrapped_result; // Should return 1 + 2 = 3
}

// CHECK-LABEL: define {{.*}} @generic_comprehensive
int generic_comprehensive(int __ob_wrap w, int __ob_trap t, int plain) {
  // CHECK: store i32 111, ptr %w_val
  int w_val = _Generic(w,
    int: 100,
    int __ob_wrap: 111,
    int __ob_trap: 222,
    char: 333,
    default: 999);

  // CHECK: store i32 222, ptr %t_val
  int t_val = _Generic(t,
    int: 100,
    int __ob_wrap: 111,
    int __ob_trap: 222,
    char: 333,
    default: 999);

  // CHECK: store i32 100, ptr %p_val
  int p_val = _Generic(plain,
    int: 100,
    int __ob_wrap: 111,
    int __ob_trap: 222,
    char: 333,
    default: 999);

  // CHECK: add nsw i32
  // CHECK: add nsw i32
  // CHECK: ret i32
  return w_val + t_val + p_val; // Should return 111 + 222 + 100 = 433
}

