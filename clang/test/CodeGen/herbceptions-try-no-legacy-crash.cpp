// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -std=c++26 -fherbceptions -fcxx-exceptions -fexceptions -emit-llvm -o - %s | FileCheck %s --check-prefix=MSVC
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++26 -fherbceptions -fcxx-exceptions -fexceptions -emit-llvm -o - %s | FileCheck %s --check-prefix=ITANIUM

// Regression test: a `throws` function whose body uses `try(expr)` over a
// throws-call should not crash CodeGen. The function has no noexcept(false)
// callees that can throw legacy C++ exceptions, so Sema does not set
// HerbceptionLegacyErrorValue; CodeGen must not attempt to emit the legacy
// conversion body (which would dereference a null Conv and segfault).

int callee() throws;

// A throws function wrapping a throws call in try(expr). No legacy C++
// exception can escape, so no herb.legacy.convert block should be emitted.
// MSVC-LABEL: define dso_local { { ptr, i64 }, i1 } @"?foo@@YAHXZ"()
// MSVC-NOT: herb.legacy.convert
int foo() throws {
  return try(callee());
}

// Same pattern on Itanium.
// ITANIUM-LABEL: define dso_local { { ptr, i64 }, i1 } @_Z3barv()
// ITANIUM-NOT: herb.legacy.convert
int bar() throws {
  return try(callee());
}
