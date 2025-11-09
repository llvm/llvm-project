// Tests that the modules driver properly compiles both C++20 named modules
// and Clang modules.
// This does not test importing a Clang module into a C++20 named module 
// interface unit, or vice versa, is not yet supported.
// TODO: Support imports between different module types.
// Because the std library is not available in the CI, this does not test for it.
// TODO: Add tests for the Standard library modules.

// RUN: split-file %s %t
// RUN: %clang++ -std=c++23 -fmodules -fmodules-driver \
// RUN:   -fmodule-map-file=%t/module.modulemap %t/main.cpp \
// RUN:   %t/A.cpp %t/A-part1.cpp %t/A-part1-impl.cpp %t/A-part2.cpp \
// RUN:   %t/B.cppm

//--- main.cpp
#include "root.h"
import A;
import B;

int main()  {
 // *** Testing C++20 named modules ***
 A();       // From the A's primary module partition interface .
 APart1();  // From a public module partition interface unit of A.

 // *** Testing Clang modules ***
 theAnswer();
}

//--- A.cpp
export module A;
export import :part1;
import :part2;
import B;

export int A() {
  doesNothing(); // Imported from B
  return APart1() + APart2(); // From public and private module partition interface units.
}

//--- A-part1.cpp
export module A:part1;

export int APart1(); // Implemented in module implementation unit A-part1-impl.cpp.

//--- A-part1-impl.cpp
module A:part1_impl;
import :part2;

int APart1() {
  return 2 + APart2();
}

//--- A-part2.cpp
export module A:part2;

export int APart2() {
  return 2;
}

//--- B.cppm
export module B;

export void doesNothing() {
  return;
}

//--- module.modulemap
module root { header "root.h" export *}
module direct1 { header "direct1.h" export *}
module direct2 { header "direct2.h" export *}
module transitive1 { header "transitive1.h" export *}
module transitive2 { header "transitive2.h" export * }

//--- root.h
#include "direct1.h"
#include "direct2.h"
int theAnswer() {
  return fromDirect1() + fromDirect2();
}

//--- direct1.h
#include "transitive1.h"
#include "transitive2.h"

int fromDirect1() {
  return fromTransitive1() + fromTransitive2();
}

//--- direct2.h
#include "transitive1.h"

int fromDirect2() {
  return fromTransitive1() + 2;
}

//--- transitive1.h
int fromTransitive1() {
  return 10;
}

//--- transitive2.h
int fromTransitive2() {
  return 10;
}
