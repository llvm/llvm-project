// RUN: %clang_analyze_cc1 -Wno-conversion -Wno-tautological-constant-compare -analyzer-checker=core,alpha.core.Conversion -verify %s

// expected-no-diagnostics

void dontwarn1() {
  unsigned long x = static_cast<unsigned long>(-1);
}

void dontwarn2(unsigned x) {
  if (x == static_cast<unsigned>(-1)) {
  }
}

struct C {
  C(unsigned x, unsigned long y) {}
};

void f(C) {}

void functioncall1(long x) {
  f(C(64, x));
}
