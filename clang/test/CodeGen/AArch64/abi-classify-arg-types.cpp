// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-abi darwinpcs -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-abi darwinpcs -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple arm64_32-apple-ios7.0 -target-abi darwinpcs -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64_32-apple-ios7.0 -target-abi darwinpcs -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64-pc-windows-msvc -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-pc-windows-msvc -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --implicit-check-not="not yet implemented"

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

}
