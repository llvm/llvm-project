// RUN: split-file %s %t

// A musttail call replaces this function's frame, and with it the handler that
// enforces the function's exception specification. Each case below needs its
// own translation unit because code generation stops at the first error.

//--- terminate_scope.cpp

// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fcxx-exceptions \
// RUN:   -fexceptions -fclangir -emit-cir -verify %t/terminate_scope.cpp -o %t/terminate_scope.cir
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fcxx-exceptions \
// RUN:   -fexceptions -emit-llvm -verify %t/terminate_scope.cpp -o %t/terminate_scope.ll

// The handler of a terminate scope calls std::terminate() when an exception
// tries to escape. Losing it along with the frame is only safe if the callee
// cannot throw either, because then the callee's own handler stands in for it.

void mayThrow();

void terminateScope() noexcept {
  // expected-error@+1 {{'musttail' in a noexcept function requires a noexcept callee}}
  [[clang::musttail]] return mayThrow();
}

//--- dynamic_spec.cpp

// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-linux-gnu -fcxx-exceptions \
// RUN:   -fexceptions -fclangir -emit-cir -verify %t/dynamic_spec.cpp -o %t/dynamic_spec.cir
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-linux-gnu -fcxx-exceptions \
// RUN:   -fexceptions -emit-llvm -verify %t/dynamic_spec.cpp -o %t/dynamic_spec.ll

// The filter of a dynamic exception specification has to run while an exception
// is leaving the function. A tail call deallocates this function's frame before
// the callee starts running, so an exception thrown by the callee unwinds into
// this function's caller. The frame holding the filter is no longer on the
// stack for the unwinder to find. Only the normal return path could be a tail
// call, which is why LLVM allows the musttail marker on call but not on invoke.
// So this is not waiting on an implementation, and the "yet" in the message is
// misleading. Sema rejects the comparable cases, such as a musttail return from
// inside a try block, outright.

void mayThrow();

void dynamicSpec() throw(int) {
  // expected-error@+1 {{cannot compile this tail call skipping over cleanups yet}}
  [[clang::musttail]] return mayThrow();
}

//--- dynamic_spec_nothrow_callee.cpp

// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-linux-gnu -fcxx-exceptions \
// RUN:   -fexceptions -fclangir -emit-cir -verify %t/dynamic_spec_nothrow_callee.cpp -o %t/dynamic_spec_nothrow_callee.cir
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-linux-gnu -fcxx-exceptions \
// RUN:   -fexceptions -emit-llvm -verify %t/dynamic_spec_nothrow_callee.cpp -o %t/dynamic_spec_nothrow_callee.ll

// The same rejection for a callee that cannot throw, where the filter could
// never run: an exception leaving the callee's body meets the callee's own
// terminate scope first, just as it would without the tail call, so nothing
// would have to be skipped. Classic codegen rejects it anyway, because a
// terminate scope is the only exception specification its musttail check
// recognizes, and CIR follows it rather than accepting what classic rejects.

void cannotThrow() noexcept;

void dynamicSpec() throw(int) {
  // expected-error@+1 {{cannot compile this tail call skipping over cleanups yet}}
  [[clang::musttail]] return cannotThrow();
}
