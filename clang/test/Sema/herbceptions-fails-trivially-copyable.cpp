// RUN: not %clang_cc1 -std=c++26 -fherbceptions -fcxx-exceptions -fsyntax-only %s 2>&1 | FileCheck %s

// The 'return_failure{E}' error type must be trivially copyable, matching C behavior:
// the error value flows through the {T, i1} ABI slot by value. Non-trivially
// copyable error types (classes with user-defined copy/move/destructor) are
// rejected. 'return_failure{E}' supersedes 'noexcept': the two cannot be combined.

struct NotTriviallyCopyable {
  NotTriviallyCopyable(const NotTriviallyCopyable&);
  ~NotTriviallyCopyable();
};

// CHECK: error: the 'return_failure{...}' error type 'NotTriviallyCopyable' must be trivially copyable
int bad() return_failure{NotTriviallyCopyable}; // expected-error {{the 'return_failure{...}' error type 'NotTriviallyCopyable' must be trivially copyable}}

// CHECK: error: the 'return_failure{...}' error type 'NotTriviallyCopyable' must be trivially copyable
int bad2() return_failure{NotTriviallyCopyable}; // expected-error {{the 'return_failure{...}' error type 'NotTriviallyCopyable' must be trivially copyable}}

// Trivially copyable error types are accepted.
int ok1() return_failure{int};

// return_failure and noexcept cannot be combined.
int bad3() return_failure{int} noexcept(false); // expected-error {{'throws'/'return_failure' and 'noexcept' cannot be combined}}
int bad4() noexcept(false) return_failure{int}; // expected-error {{'throws'/'return_failure' and 'noexcept' cannot be combined}}
int bad5() return_failure{int} noexcept(true); // expected-error {{'throws'/'return_failure' and 'noexcept' cannot be combined}}
