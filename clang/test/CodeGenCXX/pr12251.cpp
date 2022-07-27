// RUN: %clang_cc1 -no-opaque-pointers %s -triple i386-unknown-unknown -emit-llvm -O1 -relaxed-aliasing -fstrict-enums -std=c++11 -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers %s -triple i386-unknown-unknown -emit-llvm -O1 -relaxed-aliasing -std=c++11 -o - | FileCheck --check-prefix=NO-STRICT-ENUMS %s

bool f(bool *x) {
  return *x;
}
// CHECK-LABEL: define{{.*}} zeroext i1 @_Z1fPb
// CHECK: load i8, i8* %{{[^ ]*}}, align 1, !range [[RANGE_i8_0_2:![^ ]*]]

// Only enum-tests follow. Ensure that after the bool test, no further range
// metadata shows up when strict enums are disabled.
// NO-STRICT-ENUMS-LABEL: define{{.*}} zeroext i1 @_Z1fPb
// NO-STRICT-ENUMS: load i8, i8* %{{[^ ]*}}, align 1, !range
// NO-STRICT-ENUMS-NOT: !range

enum e1 { };
e1 g1(e1 *x) {
  return *x;
}
// CHECK-LABEL: define{{.*}} i32 @_Z2g1P2e1
// CHECK: ret i32 %0

enum e2 { e2_a = 0 };
e2 g2(e2 *x) {
  return *x;
}
// CHECK-LABEL: define{{.*}} i32 @_Z2g2P2e2
// CHECK: ret i32 %0

enum e3 { e3_a = 16 };
e3 g3(e3 *x) {
  return *x;
}
// CHECK-LABEL: define{{.*}} i32 @_Z2g3P2e3
// CHECK: load i32, i32* %x, align 4, !range [[RANGE_i32_0_32:![^ ]*]]

enum e4 { e4_a = -16};
e4 g4(e4 *x) {
  return *x;
}
// CHECK-LABEL: define{{.*}} i32 @_Z2g4P2e4
// CHECK: load i32, i32* %x, align 4, !range [[RANGE_i32_m16_16:![^ ]*]]

enum e5 { e5_a = -16, e5_b = 16};
e5 g5(e5 *x) {
  return *x;
}
// CHECK-LABEL: define{{.*}} i32 @_Z2g5P2e5
// CHECK: load i32, i32* %x, align 4, !range [[RANGE_i32_m32_32:![^ ]*]]

enum e6 { e6_a = -1 };
e6 g6(e6 *x) {
  return *x;
}
// CHECK-LABEL: define{{.*}} i32 @_Z2g6P2e6
// CHECK: load i32, i32* %x, align 4, !range [[RANGE_i32_m1_1:![^ ]*]]

enum e7 { e7_a = -16, e7_b = 2};
e7 g7(e7 *x) {
  return *x;
}
// CHECK-LABEL: define{{.*}} i32 @_Z2g7P2e7
// CHECK: load i32, i32* %x, align 4, !range [[RANGE_i32_m16_16]]

enum e8 { e8_a = -17};
e8 g8(e8 *x) {
  return *x;
}
// CHECK-LABEL: define{{.*}} i32 @_Z2g8P2e8
// CHECK: load i32, i32* %x, align 4, !range [[RANGE_i32_m32_32:![^ ]*]]

enum e9 { e9_a = 17};
e9 g9(e9 *x) {
  return *x;
}
// CHECK-LABEL: define{{.*}} i32 @_Z2g9P2e9
// CHECK: load i32, i32* %x, align 4, !range [[RANGE_i32_0_32]]

enum e10 { e10_a = -16, e10_b = 32};
e10 g10(e10 *x) {
  return *x;
}
// CHECK-LABEL: define{{.*}} i32 @_Z3g10P3e10
// CHECK: load i32, i32* %x, align 4, !range [[RANGE_i32_m64_64:![^ ]*]]

enum e11 {e11_a = 4294967296 };
enum e11 g11(enum e11 *x) {
  return *x;
}
// CHECK-LABEL: define{{.*}} i64 @_Z3g11P3e11
// CHECK: load i64, i64* %x, align {{[84]}}, !range [[RANGE_i64_0_2pow33:![^ ]*]]

enum e12 {e12_a = 9223372036854775808U };
enum e12 g12(enum e12 *x) {
  return *x;
}
// CHECK-LABEL: define{{.*}} i64 @_Z3g12P3e12
// CHECK: load i64, i64* %x, align {{[84]}}
// CHECK-NOT: range
// CHECK: ret

enum e13 : char {e13_a = -1 };
e13 g13(e13 *x) {
  return *x;
}
// CHECK-LABEL: define{{.*}} signext i8 @_Z3g13P3e13
// CHECK: load i8, i8* %x, align 1
// CHECK-NOT: range
// CHECK: ret

enum class e14 {e14_a = 1};
e14 g14(e14 *x) {
  return *x;
}
// CHECK-LABEL: define{{.*}} i32 @_Z3g14P3e14
// CHECK: load i32, i32* %x, align 4
// CHECK-NOT: range
// CHECK: ret

enum e15 { e15_a = 2147483648 };
e15 g15(e15 *x) {
  return *x;
}
// CHECK-LABEL: define{{.*}} i32 @_Z3g15P3e15
// CHECK: load i32, i32* %x, align 4
// CHECK-NOT: range
// CHECK: ret

enum e16 { e16_a = -2147483648 };
e16 g16(e16 *x) {
  return *x;
}
// CHECK-LABEL: define{{.*}} i32 @_Z3g16P3e16
// CHECK: load i32, i32* %x, align 4
// CHECK-NOT: range
// CHECK: ret


// CHECK: [[RANGE_i8_0_2]] = !{i8 0, i8 2}
// CHECK: [[RANGE_i32_0_32]] = !{i32 0, i32 32}
// CHECK: [[RANGE_i32_m16_16]] = !{i32 -16, i32 16}
// CHECK: [[RANGE_i32_m32_32]] = !{i32 -32, i32 32}
// CHECK: [[RANGE_i32_m1_1]] = !{i32 -1, i32 1}
// CHECK: [[RANGE_i32_m64_64]] = !{i32 -64, i32 64}
// CHECK: [[RANGE_i64_0_2pow33]] = !{i64 0, i64 8589934592}
