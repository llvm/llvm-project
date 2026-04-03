// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm -o - %s -target-feature +avx | FileCheck %s --check-prefix=SYSV
// RUN: %clang_cc1 -triple x86_64-sie-ps5 -std=c++20 -emit-llvm -o - %s -target-feature +avx | FileCheck %s --check-prefix=PS
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclang-abi-compat=22 -emit-llvm -o - %s -target-feature +avx | FileCheck %s --check-prefix=CLANG22

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

// Both the empty base case and empty-field case are now passed in register as per SysV spec
// SYSV-LABEL: define dso_local <4 x i64> @_Z29return_empty_base_then_vectorv()
// SYSV-LABEL: define dso_local noundef i64 @_Z27pass_empty_base_then_vector19EmptyBaseThenVector(<4 x i64> %X.coerce)
// SYSV-LABEL: define dso_local <4 x i64> @_Z30return_empty_field_then_vectorv()
// SYSV-LABEL: define dso_local noundef i64 @_Z28pass_empty_field_then_vector20EmptyFieldThenVector(<4 x i64> %X.coerce)

// PlayStation keeps the legacy ABI behavior here: the empty-base case was always direct for PlayStation ABI
// while the [[no_unique_address]] empty-field case is still indirect even after this change.
// PS-LABEL: define dso_local <4 x i64> @_Z29return_empty_base_then_vectorv()
// PS-LABEL: define dso_local noundef i64 @_Z27pass_empty_base_then_vector19EmptyBaseThenVector(<4 x i64> %X.coerce)
// PS-LABEL: define dso_local void @_Z30return_empty_field_then_vectorv(ptr dead_on_unwind noalias writable sret(%struct.EmptyFieldThenVector) align 32 %agg.result)
// PS-LABEL: define dso_local noundef i64 @_Z28pass_empty_field_then_vector20EmptyFieldThenVector(ptr noundef byval(%struct.EmptyFieldThenVector) align 32 %X)

// Clang22 ABI compatibility mode keeps the legacy ABI behavior: the empty-base case and [[no_unique_address]] empty-field case both as indirect
// CLANG22-LABEL: define dso_local void @_Z29return_empty_base_then_vectorv(ptr dead_on_unwind noalias writable sret(%struct.EmptyBaseThenVector) align 32 %agg.result) 
// CLANG22-LABEL: define dso_local noundef i64 @_Z27pass_empty_base_then_vector19EmptyBaseThenVector(ptr noundef byval(%struct.EmptyBaseThenVector) align 32 %X)
// CLANG22-LABEL: define dso_local void @_Z30return_empty_field_then_vectorv(ptr dead_on_unwind noalias writable sret(%struct.EmptyFieldThenVector) align 32 %agg.result)
// CLANG22-LABEL: define dso_local noundef i64 @_Z28pass_empty_field_then_vector20EmptyFieldThenVector(ptr noundef byval(%struct.EmptyFieldThenVector) align 32 %X)
