// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-abi darwinpcs -std=c++20 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-abi darwinpcs -std=c++20 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64-linux-gnu -std=c++20 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -std=c++20 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --implicit-check-not="not yet implemented"

// Verify C++ homogeneous floating-point aggregate return classification matches
// between classic CodeGen and the LLVM ABI library.

struct HFA2f {
  float a, b;
};
HFA2f ret_hfa2f() { return {}; }
// CHECK: define{{.*}} %struct.HFA2f @_Z9ret_hfa2fv()

struct EmptyBase {};
struct HFAEmptyBase : EmptyBase {
  float a, b;
};
HFAEmptyBase ret_hfa_empty_base() { return {}; }
// CHECK: define{{.*}} %struct.HFAEmptyBase @_Z18ret_hfa_empty_basev()

struct EmptyBase1 {};
struct EmptyBase2 {};
struct HFAMultiEmptyBase : EmptyBase1, EmptyBase2 {
  float a, b;
};
HFAMultiEmptyBase ret_hfa_multi_empty_base() { return {}; }
// CHECK: define{{.*}} %struct.HFAMultiEmptyBase @_Z24ret_hfa_multi_empty_basev()

struct HFABase {
  float a;
};
struct HFADerived : HFABase {
  float b;
};
HFADerived ret_hfa_derived() { return {}; }
// CHECK: define{{.*}} %struct.HFADerived @_Z15ret_hfa_derivedv()

struct HFABaseAndFields : HFABase {
  float b, c;
};
HFABaseAndFields ret_hfa_base_fields() { return {}; }
// CHECK: define{{.*}} %struct.HFABaseAndFields @_Z19ret_hfa_base_fieldsv()

struct FloatBase1 {
  float a;
};
struct FloatBase2 {
  float b;
};
struct HFATwoBases : FloatBase1, FloatBase2 {};
HFATwoBases ret_hfa_two_bases() { return {}; }
// CHECK: define{{.*}} %struct.HFATwoBases @_Z17ret_hfa_two_basesv()

struct HFANested {
  HFA2f inner;
  float c;
};
HFANested ret_hfa_nested() { return {}; }
// CHECK: define{{.*}} %struct.HFANested @_Z14ret_hfa_nestedv()

struct HFAZeroBF {
  int : 0;
  float a, b;
};
HFAZeroBF ret_hfa_zerobf() { return {}; }
// CHECK: define{{.*}} %struct.HFAZeroBF @_Z14ret_hfa_zerobfv()

struct Empty {};
struct HFANoUniqueEmpty {
  [[no_unique_address]] Empty e;
  float a, b;
};
HFANoUniqueEmpty ret_hfa_nua_empty() { return {}; }
// CHECK: define{{.*}} %struct.HFANoUniqueEmpty @_Z17ret_hfa_nua_emptyv()

_Complex float ret_complex_float() { return 1.0f; }
// CHECK: define{{.*}} { float, float } @_Z17ret_complex_floatv()
