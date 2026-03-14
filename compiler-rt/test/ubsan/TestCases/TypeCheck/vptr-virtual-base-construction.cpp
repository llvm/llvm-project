// RUN: %clangxx -frtti -fsanitize=vptr -fno-sanitize-recover=vptr %s -o %t
// RUN: %run %t

// REQUIRES: cxxabi
// FIXME: Should pass on Android, but started failing around 2023-11-05 for unknown reasons.
// UNSUPPORTED: target={{.*(windows-msvc|android).*}}

int volatile n;

struct A { virtual ~A() {} };
struct B: virtual A {};
struct C: virtual A { ~C() { n = 0; } };
struct D: virtual B, virtual C {};

int main() { delete new D; }
