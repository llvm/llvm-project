// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --input-file=%t.og.ll %s --check-prefix=OGCG
//
// XFAIL: *
//
// CIR→LLVM lowering does not implement sret (struct return) calling convention.
//
// When returning non-trivial C++ types by value on x86_64, the System V ABI
// requires using the sret calling convention:
// - The function should return void
// - A hidden first parameter (ptr sret) receives the return value address
// - The caller allocates space for the return value
//
// Currently, CIR→LLVM lowering returns structs by value directly, which is
// not ABI-compliant and will cause calling convention mismatches with code
// compiled via standard CodeGen.
//
// This affects any function returning a non-trivial struct/class by value.

struct S {
  int x;
  ~S();  // Non-trivial destructor makes this sret-eligible
};

S foo() {
  S s;
  s.x = 42;
  return s;
}

// LLVM lowering incorrectly returns %struct.S by value
// LLVM: define dso_local %struct.S @_Z3foov()
// LLVM-NOT: sret

// Original CodeGen correctly uses sret calling convention
// OGCG: define dso_local void @_Z3foov(ptr {{.*}}sret(%struct.S){{.*}} %agg.result)
// OGCG-NOT: define {{.*}} %struct.S @_Z3foov

// Expected LLVM lowering (when fixed):
// Should match OGCG: define dso_local void @_Z3foov(ptr sret(%struct.S) %agg.result)
