// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm -o - %s -target-feature +avx | FileCheck %s --check-prefix=SYSV
// RUN: %clang_cc1 -triple x86_64-sie-ps5 -std=c++20 -emit-llvm -o - %s -target-feature +avx | FileCheck %s --check-prefix=PS
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclang-abi-compat=23 -emit-llvm -o - %s -target-feature +avx | FileCheck %s --check-prefix=CLANG22

typedef unsigned long long v4ull __attribute__((vector_size(32)));

struct EmptyBase {};

struct EmptyBaseThenVector : EmptyBase {
  v4ull Data;
};

EmptyBaseThenVector return_empty_base_then_vector() {
  return {};
}

unsigned long long pass_empty_base_then_vector(EmptyBaseThenVector X) {
  return X.Data[0];
}

struct EmptyField {};

struct EmptyFieldThenVector {
  [[no_unique_address]] EmptyField E;
  v4ull Data;
};

EmptyFieldThenVector return_empty_field_then_vector() {
  return {};
}

unsigned long long pass_empty_field_then_vector(EmptyFieldThenVector X) {
  return X.Data[0];
}

// Empty base classes are covered by CodeGen/X86/avx-cxx-record.cpp. This test
// covers [[no_unique_address]] empty fields, which are not handled by the
// legacy classifier.
// SYSV-LABEL: define dso_local <4 x i64> @_Z30return_empty_field_then_vectorv()
// SYSV-LABEL: define dso_local noundef i64 @_Z28pass_empty_field_then_vector20EmptyFieldThenVector(<4 x i64> %X.coerce)

// PlayStation keeps the legacy ABI behavior for both cases.
// PS-LABEL: define dso_local <4 x i64> @_Z29return_empty_base_then_vectorv()
// PS-LABEL: define dso_local noundef i64 @_Z27pass_empty_base_then_vector19EmptyBaseThenVector(<4 x i64> %X.coerce)
// PS-LABEL: define dso_local void @_Z30return_empty_field_then_vectorv(ptr dead_on_unwind noalias writable sret(%struct.EmptyFieldThenVector) align 32 %agg.result)
// PS-LABEL: define dso_local noundef i64 @_Z28pass_empty_field_then_vector20EmptyFieldThenVector(ptr noundef byval(%struct.EmptyFieldThenVector) align 32 %X)

// Clang 22 ABI compatibility mode keeps the legacy ABI behavior for
// [[no_unique_address]] empty fields.
// CLANG22-LABEL: define dso_local void @_Z30return_empty_field_then_vectorv(ptr dead_on_unwind noalias writable sret(%struct.EmptyFieldThenVector) align 32 %agg.result)
// CLANG22-LABEL: define dso_local noundef i64 @_Z28pass_empty_field_then_vector20EmptyFieldThenVector(ptr noundef byval(%struct.EmptyFieldThenVector) align 32 %X)
