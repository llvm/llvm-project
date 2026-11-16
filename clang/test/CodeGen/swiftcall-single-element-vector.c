// RUN: %clang_cc1 -no-enable-noundef-analysis -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,X86
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple arm64-apple-ios9 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,ARM,ARM64
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple armv7-apple-darwin9 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,ARM

#define SWIFTCALL __attribute__((swiftcall))

typedef long long long1 __attribute__((vector_size(8)));
typedef double double1 __attribute__((ext_vector_type(1)));
typedef float float1 __attribute__((ext_vector_type(1)));

// A 64-bit single-element vector is legal on ARM, but must be scalarized on x86.
// X86-LABEL: define swiftcc i64 @pass_long1(i64
// X86: ret i64
// ARM-LABEL: define swiftcc <1 x i64> @pass_long1(<1 x i64>
// ARM: ret <1 x i64>
SWIFTCALL long1 pass_long1(long1 value) {
  return value;
}

// X86-LABEL: define swiftcc double @pass_double1(double
// X86: ret double
// ARM-LABEL: define swiftcc <1 x double> @pass_double1(<1 x double>
// ARM: ret <1 x double>
SWIFTCALL double1 pass_double1(double1 value) {
  return value;
}

// A 32-bit vector must be scalarized on all three targets.
// CHECK-LABEL: define swiftcc float @pass_float1(float
// CHECK: ret float
SWIFTCALL float1 pass_float1(float1 value) {
  return value;
}

struct VectorBox {
  long1 v;
};

// X86-LABEL: define swiftcc i64 @pass_box(i64
// X86: ret i64
// ARM-LABEL: define swiftcc <1 x i64> @pass_box(<1 x i64>
// ARM: ret <1 x i64>
SWIFTCALL struct VectorBox pass_box(struct VectorBox value) {
  return value;
}

// CHECK-LABEL: define {{.*}} @call_pass_box(
// X86: call swiftcc i64 @pass_box(i64
// ARM: call swiftcc <1 x i64> @pass_box(<1 x i64>
struct VectorBox call_pass_box(struct VectorBox value) {
  return pass_box(value);
}

#ifdef __SIZEOF_INT128__
typedef __int128 int128_1 __attribute__((vector_size(16)));

// A 128-bit single-element vector is legal on x86, but not on ARM64.
// X86-LABEL: define swiftcc <1 x i128> @pass_int128_1(<1 x i128>
// X86: ret <1 x i128>
// ARM64-LABEL: define swiftcc i128 @pass_int128_1(i128
// ARM64: ret i128
SWIFTCALL int128_1 pass_int128_1(int128_1 value) {
  return value;
}
#endif
