// RUN: %clang_cc1 -triple x86_64-linux-gnu -foverflow-behavior-types -std=c2x %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK

#define __wrap __attribute__((overflow_behavior(wrap)))
#define __nowrap __attribute__((overflow_behavior(no_wrap)))

typedef int __wrap w_int;
typedef int __nowrap nw_int;

// CHECK-LABEL: define {{.*}} @generic_selection_test_nomatch
int generic_selection_test_nomatch(int x) {
  // CHECK: ret i32 3
  return _Generic(x, w_int: 1, nw_int: 2, default: 3);
}

// CHECK-LABEL: define {{.*}} @generic_selection_test_obtmatch
int generic_selection_test_obtmatch(w_int x) {
  // CHECK: ret i32 1
  return _Generic(x, w_int: 1, nw_int: 2, default: 3);
}

// CHECK-LABEL: define {{.*}} @signed_bitint_test
void signed_bitint_test(_BitInt(4) __nowrap a, _BitInt(8) __nowrap b, _BitInt(99) __wrap c) {
  // CHECK: call { i4, i1 } @llvm.sadd.with.overflow.i4(i4
  (a + 1);

  // CHECK: call { i8, i1 } @llvm.sadd.with.overflow.i8(i8
  (b + 1);

  // CHECK: add i99 {{.*}}, 1
  (c + 1);
}

// CHECK-LABEL: define {{.*}} @unsigned_bitint_test
void unsigned_bitint_test(unsigned _BitInt(4) __nowrap a, unsigned _BitInt(8) __nowrap b, unsigned _BitInt(99) __wrap c) {
  // CHECK: call { i4, i1 } @llvm.uadd.with.overflow.i4(i4
  (a + 1);

  // CHECK: call { i8, i1 } @llvm.uadd.with.overflow.i8(i8
  (b + 1);

   // CHECK: add i99 {{.*}}, 1
  (c + 1);
}

