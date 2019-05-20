// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -mllvm -warn-stack-size=0 -emit-codegen-only -triple=i386-apple-darwin 2>&1 | FileCheck %s

// TODO: Emit rich diagnostics for thunks and move this into the appropriate test file.
// Until then, test that we fall back and display the LLVM backend diagnostic.
namespace frameSizeThunkWarning {
  struct A {
    virtual void f();
  };

  struct B : virtual A {
    virtual void f();
  };

  // CHECK: warning: stack frame size of {{[0-9]+}} bytes in function 'frameSizeThunkWarning::B::f'
  // CHECK: warning: stack size limit exceeded ({{[0-9]+}}) in {{[^ ]+}}
  void B::f() { }
}
