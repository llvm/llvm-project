// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -fwarn-stack-size=0 -emit-codegen-only -triple=i386-apple-darwin 2>&1 | FileCheck %s

// TODO: Emit rich diagnostics for thunks and move this into the appropriate test file.
// Until then, test that we fall back and display the LLVM backend diagnostic.
namespace frameSizeThunkWarning {
  struct A {
    virtual void f();
  };

  struct B : virtual A {
    virtual void f();
  };

  // CHECK: warning: stack frame size ([[#]]) exceeds limit ([[#]]) in 'frameSizeThunkWarning::B::f()'
  // CHECK: warning: stack frame size ([[#]]) exceeds limit ([[#]]) in function '_ZTv0_n12_N21frameSizeThunkWarning1B1fEv'
  void B::f() {
    volatile int x = 0; // Ensure there is stack usage.
  }
}
