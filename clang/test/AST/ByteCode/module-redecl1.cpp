// RUN: %clang_cc1 -cc1 -xobjective-c++ %s -fmodules -fimplicit-module-maps -fmodules-cache-path=Inputs/ -I %S/Inputs -verify -std=c++11
// RUN: %clang_cc1 -cc1 -xobjective-c++ %s -fmodules -fimplicit-module-maps -fmodules-cache-path=Inputs/ -I %S/Inputs -verify -std=c++11 -fexperimental-new-constant-interpreter

// expected-no-diagnostics

struct S { int a; };

extern const int variable;
extern const S vars;

constexpr int test() { return variable; }
constexpr int test2() { return vars.a; }

struct C {
  static const int variable;
  static const S vars;
};



/// The module contains a definition for 'variable', so the function call below
/// should work and return the correct value.
@import redecl1;
static_assert(test() == 120, "");
static_assert(test2() == 12, "");
