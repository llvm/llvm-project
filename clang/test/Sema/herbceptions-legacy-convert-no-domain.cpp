// RUN: %clang_cc1 -std=c++26 -fherbceptions -fcxx-exceptions -fexceptions -fsyntax-only -verify %s
// expected-no-diagnostics

// A `throws` function calling a noexcept(false) callee: the compiler fabricates
// the legacy-exception-to-herbception conversion using built-in ABI calls.
// No std::exception_ptr / std::error / error_domain declarations needed.

extern void legacy_callee() noexcept(false);

void calls_legacy() throws {
  legacy_callee();
}
