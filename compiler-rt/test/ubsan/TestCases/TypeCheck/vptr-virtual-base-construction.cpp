// RUN: %clangxx -frtti -fsanitize=vptr -fno-sanitize-recover=vptr %s -o %t
// RUN: %run %t

// REQUIRES: cxxabi
// UNSUPPORTED: windows-msvc

int volatile n;

struct A { virtual ~A() {} };
struct B: virtual A {};
struct C: virtual A { ~C() { n = 0; } };
struct D: virtual B, virtual C {};

int main() { delete new D; }
