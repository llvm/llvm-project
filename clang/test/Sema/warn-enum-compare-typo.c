// RUN: %clang_cc1 -fsyntax-only -Wenum-compare-typo -verify %s 
// RUN: %clang_cc1 -fsyntax-only -Wenum-compare-typo -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

enum E {
  kOptionGood1 = 1ull << 0,
  kOptionGood2 = 1ull >> 0,
  // expected-warning@+3 {{comparison operator '<' in enumerator constant is likely a typo for the shift operator '<<'}} 
  // expected-note@+2 {{use '<<' to perform a bitwise shift}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:22-[[@LINE+1]]:23}:"<<"
  kOptionBad1 = 1ull < 1,
  // expected-warning@+3 {{comparison operator '>' in enumerator constant is likely a typo for the shift operator '>>'}}
  // expected-note@+2 {{use '>>' to perform a bitwise shift}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:19-[[@LINE+1]]:20}:">>"
  kOptionBad2 = 1 > 3,
  // expected-warning@+3 {{comparison operator '>' in enumerator constant is likely a typo for the shift operator '>>'}} 
  // expected-note@+2 {{use '>>' to perform a bitwise shift}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:20-[[@LINE+1]]:21}:">>"
  kOptionBad3 = (1 > 2)
};

// Ensure the warning does not fire on valid code
enum F {
  kSomeValue = 10,
  kComparison = kSomeValue > 5, // No warning
  kMaxEnum = 255,
  kIsValid = kMaxEnum > 0, // No warning
  kSum = 10 + 20,          // No warning
  kShift = 2 << 5          // No warning
};

// Ensure the diagnostic group works

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wenum-compare-typo"
enum G {
  kIgnored = 1 < 10 // No warning
};
#pragma clang diagnostic pop
