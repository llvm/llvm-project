// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-abi darwinpcs -std=c++20 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,NOHFAALIGN,GPR64
// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-abi darwinpcs -std=c++20 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,NOHFAALIGN,GPR64 --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple arm64_32-apple-ios7.0 -target-abi darwinpcs -std=c++20 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,NOHFAALIGN,ILP32
// RUN: %clang_cc1 -triple arm64_32-apple-ios7.0 -target-abi darwinpcs -std=c++20 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,NOHFAALIGN,ILP32 --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64-linux-gnu -std=c++20 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,AAPCS64,GPR64
// RUN: %clang_cc1 -triple aarch64-linux-gnu -std=c++20 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AAPCS64,GPR64 --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -std=c++20 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,AAPCS64,GPR64
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -std=c++20 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AAPCS64,GPR64 --implicit-check-not="not yet implemented"

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

// Aggregates of at most 16 bytes are passed directly. A one-byte empty class
// is ignored on Darwin and passed as i64 on AAPCS. An empty base makes the
// record an integer slot instead of a pointer.

struct Empty {};
void arg_empty_record(Empty e) {}
// AAPCS64: define{{.*}} void @arg_empty_record(i64 %{{.*}})
// NOHFAALIGN: define{{.*}} void @arg_empty_record()

struct OnePtr {
  void *p;
};
struct NestedPtr {
  OnePtr inner;
};
void arg_nested_ptr(NestedPtr s) {}
// GPR64: define{{.*}} void @arg_nested_ptr(ptr %{{.*}})
// ILP32: define{{.*}} void @arg_nested_ptr(i32 %{{.*}})

struct PtrBase {
  void *a;
};
struct DerivedPtr : PtrBase {
  void *b;
};
void arg_derived_ptr(DerivedPtr d) {}
// GPR64: define{{.*}} void @arg_derived_ptr([2 x ptr] %{{.*}})
// ILP32: define{{.*}} void @arg_derived_ptr([2 x i32] %{{.*}})

struct EmptyBase {};
struct PtrWithEmptyBase : EmptyBase {
  void *p;
};
void arg_ptr_empty_base(PtrWithEmptyBase s) {}
// GPR64: define{{.*}} void @arg_ptr_empty_base(i64 %{{.*}})
// ILP32: define{{.*}} void @arg_ptr_empty_base(i32 %{{.*}})

struct __attribute__((aligned(16))) OveralignedInt {
  int a;
};
void arg_overaligned_int(OveralignedInt s) {}
// AAPCS64: define{{.*}} void @arg_overaligned_int([2 x i64] %{{.*}})
// NOHFAALIGN: define{{.*}} void @arg_overaligned_int(i128 %{{.*}})

struct __attribute__((aligned(16))) OveralignedPtr {
  void *p;
};
void arg_overaligned_ptr(OveralignedPtr s) {}
// AAPCS64: define{{.*}} void @arg_overaligned_ptr([2 x ptr] %{{.*}})
// NOHFAALIGN: define{{.*}} void @arg_overaligned_ptr(i128 %{{.*}})

struct ThreeInts {
  int a, b, c;
};
void arg_three_ints(ThreeInts s) {}
// GPR64: define{{.*}} void @arg_three_ints([2 x i64] %{{.*}})
// ILP32: define{{.*}} void @arg_three_ints([3 x i32] %{{.*}})

}

// C++ empty records occupy a byte and are not ignored as AAPCS arguments.
// GNU zero-length arrays produce sizeof == 0, which is ignored in every
// AArch64 ABI, including C++ AAPCS.
struct ZeroSize {
  int arr[0];
};
struct NestedZeroSize {
  ZeroSize inner;
};

extern "C" {
void arg_zerosize(ZeroSize z) {}
// CHECK: define{{.*}} void @arg_zerosize()

void arg_nested_zerosize(NestedZeroSize z) {}
// CHECK: define{{.*}} void @arg_nested_zerosize()
}
