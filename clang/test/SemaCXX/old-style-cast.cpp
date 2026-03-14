// RUN: %clang_cc1 -triple x86_64-apple-darwin -fsyntax-only -verify -Wold-style-cast %s

void test1() {
  long x = (long)12; // expected-warning {{use of old-style cast}}
  (long)x; // expected-warning {{use of old-style cast}} expected-warning {{expression result unused}}
  (void**)x; // expected-warning {{use of old-style cast}} expected-warning {{expression result unused}}
  long y = static_cast<long>(12);
  (void)y;
  typedef void VOID;
  (VOID)y;
}
