// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-abi darwinpcs -std=c++20 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,NOHFAALIGN
// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-abi darwinpcs -std=c++20 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,NOHFAALIGN --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple arm64_32-apple-ios7.0 -target-abi darwinpcs -std=c++20 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,NOHFAALIGN
// RUN: %clang_cc1 -triple arm64_32-apple-ios7.0 -target-abi darwinpcs -std=c++20 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,NOHFAALIGN --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64-linux-gnu -std=c++20 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,AAPCS64
// RUN: %clang_cc1 -triple aarch64-linux-gnu -std=c++20 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AAPCS64 --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -std=c++20 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,AAPCS64
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -std=c++20 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AAPCS64 --implicit-check-not="not yet implemented"

// This test is verifying that the LLVM ABI library classifies C++ record
// arguments that cannot be passed in registers the same way Clang does without
// the library.

// Structures with a non-trivial copy constructor or destructor are passed
// indirectly (as a pointer) rather than in registers.

extern "C" {

struct NonTrivialCopy {
  NonTrivialCopy(const NonTrivialCopy &);
  int x;
};

struct NonTrivialDtorAndCopy {
  NonTrivialDtorAndCopy(const NonTrivialDtorAndCopy &);
  ~NonTrivialDtorAndCopy();
  int x;
};

struct ExplicitCopy {
  ExplicitCopy();
  ExplicitCopy(const ExplicitCopy &);
  short s;
};

void arg_nontrivial_copy(NonTrivialCopy a) {}
// CHECK: define{{.*}} void @arg_nontrivial_copy(ptr nofreeobj noundef align 4 dead_on_return dereferenceable(4) %{{.*}})

void arg_nontrivial_dtor_and_copy(NonTrivialDtorAndCopy a) {}
// CHECK: define{{.*}} void @arg_nontrivial_dtor_and_copy(ptr nofreeobj noundef align 4 {{(dead_on_return )?}}dereferenceable(4) %{{.*}})

void arg_explicit_copy(ExplicitCopy a) {}
// CHECK: define{{.*}} void @arg_explicit_copy(ptr nofreeobj noundef align 2 dead_on_return dereferenceable(2) %{{.*}})

// Homogeneous aggregates that can pass in registers are coerced to an array of
// the base type, including inherited members, nested records, and zero-length
// bitfields.

struct HFA2f {
  float a, b;
};
void arg_hfa2f(HFA2f h) {}
// AAPCS64: define{{.*}} void @arg_hfa2f([2 x float] alignstack(8) %{{.*}})
// NOHFAALIGN: define{{.*}} void @arg_hfa2f([2 x float] %{{.*}})

struct HFABase {
  float a;
};
struct HFADerived : HFABase {
  float b;
};
void arg_hfa_derived(HFADerived h) {}
// AAPCS64: define{{.*}} void @arg_hfa_derived([2 x float] alignstack(8) %{{.*}})
// NOHFAALIGN: define{{.*}} void @arg_hfa_derived([2 x float] %{{.*}})

struct HFANested {
  HFA2f inner;
  float c;
};
void arg_hfa_nested(HFANested h) {}
// AAPCS64: define{{.*}} void @arg_hfa_nested([3 x float] alignstack(8) %{{.*}})
// NOHFAALIGN: define{{.*}} void @arg_hfa_nested([3 x float] %{{.*}})

struct HFAZeroBF {
  int : 0;
  float a, b;
};
void arg_hfa_zerobf(HFAZeroBF h) {}
// AAPCS64: define{{.*}} void @arg_hfa_zerobf([2 x float] alignstack(8) %{{.*}})
// NOHFAALIGN: define{{.*}} void @arg_hfa_zerobf([2 x float] %{{.*}})

struct __attribute__((aligned(16))) OveralignedHFA {
  double a, b;
};
void arg_overaligned_hfa(OveralignedHFA h) {}
// AAPCS64: define{{.*}} void @arg_overaligned_hfa([2 x double] alignstack(8) %{{.*}})
// NOHFAALIGN: define{{.*}} void @arg_overaligned_hfa([2 x double] %{{.*}})

// A record laid out with a base class tracks unadjusted alignment the same
// way. The base and the field already fill 16 bytes, so aligned(16) adds no
// padding and the type stays homogeneous.
struct DoubleBase {
  double a;
};
struct __attribute__((aligned(16))) OveralignedDerivedHFA : DoubleBase {
  double b;
};
void arg_overaligned_derived_hfa(OveralignedDerivedHFA d) {}
// AAPCS64: define{{.*}} void @arg_overaligned_derived_hfa([2 x double] alignstack(8) %{{.*}})
// NOHFAALIGN: define{{.*}} void @arg_overaligned_derived_hfa([2 x double] %{{.*}})

}
