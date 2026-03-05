// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t-codegen.ll
// RUN: FileCheck --check-prefix=CIR --input-file=%t-cir.ll %s
// RUN: FileCheck --check-prefix=CODEGEN --input-file=%t-codegen.ll %s

// XFAIL: *

// This test documents a divergence between CIR and CodeGen:
// CIR does not emit 'unnamed_addr' attribute on vtables.
// This is a bug that needs to be fixed.
//
// Expected (CodeGen):
//   @_ZTV4Base = linkonce_odr unnamed_addr constant { [3 x ptr] } ...
//
// Actual (CIR):
//   @_ZTV4Base = linkonce_odr global { [3 x ptr] } ...
//
// The vtable should be marked as 'unnamed_addr' because:
// 1. The address of a vtable is never taken or used for identity comparison
// 2. This allows the linker to merge duplicate vtables across translation units
// 3. Reduces binary size and improves performance
// 4. CodeGen has always emitted them with this attribute

class Base {
public:
  virtual void foo() {}
};

void test() {
  Base b;
  b.foo();
}

// Both should emit unnamed_addr attribute
// CIR: @_ZTV4Base = linkonce_odr unnamed_addr
// CODEGEN: @_ZTV4Base = linkonce_odr unnamed_addr
