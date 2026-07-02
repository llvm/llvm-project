// A static-storage object whose type has a non-trivial destructor must compile
// on 32-bit ARM.  There, structors return 'this', so the implicit destructor
// call emitted for the global must be typed against the destructor's real
// (pointer-returning) signature; a void-typed call used to fail verification
// with "'cir.call' op incorrect number of results for callee".
//
// RUN: %clang_cc1 -std=c++20 -triple arm-linux-gnueabihf -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple arm-linux-gnueabihf -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=ARM --input-file=%t.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-x86.ll
// RUN: FileCheck --check-prefix=X86 --input-file=%t-x86.ll %s

struct S {
  S();
  ~S();
  int x;
};

S g;

// In CIR the destructor call for the global is built against the structor's
// real ('this'-returning) signature, so the variable initializer verifies and
// registers the destructor with __cxa_atexit.
// CIR-LABEL: cir.func internal private @__cxx_global_var_init()
// CIR: cir.call @_ZN1SC1Ev({{.*}}) : ({{.*}}) -> (!cir.ptr<!rec_S>
// CIR: cir.call @__cxa_atexit(

// On ARM the constructor returns 'this' and the non-trivial destructor is
// registered with __cxa_atexit for the global.
// ARM-LABEL: define internal void @__cxx_global_var_init()
// ARM:         call noundef ptr @_ZN1SC1Ev(ptr {{.*}} @g)
// ARM:         call i32 @__cxa_atexit(ptr @_ZN1SD1Ev, ptr @g, ptr @__dso_handle)

// On x86_64 the structors return void; registration is otherwise identical.
// X86-LABEL: define internal void @__cxx_global_var_init()
// X86:         call void @_ZN1SC1Ev(ptr {{.*}} @g)
// X86:         call i32 @__cxa_atexit(ptr @_ZN1SD1Ev, ptr @g, ptr @__dso_handle)
