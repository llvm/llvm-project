// RUN: %clang_cc1 -std=c++11 -verify %s -Wno-unused

// This must be at the start of the file (the failure depends on a SmallPtrSet
// not having been reallocated yet).
void fn1() {
  // expected-no-diagnostics
  constexpr int kIsolationClass = 0;
  const int kBytesPerConnection = 0;
  [=] { kIsolationClass, kBytesPerConnection, kBytesPerConnection; };
}
