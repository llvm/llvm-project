// Tests that the modules driver properly compiles both C++20 named modules
// and Clang modules.
// This does not test importing a Clang module into a C++20 named module 
// interface unit, or vice versa, is not yet supported.
// TODO: Support imports between different module types.

// RUN: split-file %s %t
// RUN: %clang++ -std=c++23 -fmodules -fmodules-driver \
// RUN:   -fmodule-map-file=%t/module.modulemap %t/main.cpp \
// RUN:   %t/A.cpp %t/A-B.cpp %t/A-C.cpp %t/B.cpp

//--- main.cpp
#include "root.h"
import A;
import B;

int main()  {
 sayHello();
 helloWorld();
 sayGoodbye();
 sayGoodbyeTwice();
}

//--- A.cpp
export module A;
export import :B;
import std;
import :C;


export void sayHello() {
  sayHelloImpl();
  std::println("!");
}

//--- A-B.cpp
module;
export module A:B;
import std;
import :C;

export void helloWorld() {
  sayHelloImpl();	
  std::print(" World!\n");
}

//--- A-C.cpp
export module A:C;
import std;

void sayHelloImpl() {
  std::print("Hello");
}

//--- B.cpp
module;
export module B;
import A;

export void sayHelloWorldTwice() {
  helloWorld();
  helloWorld();
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
void sayGoodbyeTwice() {
  std::println("Goodbye!");
  std::println("Goodbye!");
}

//--- direct1.h
#include "transitive1.h"
#include "transitive2.h"

//--- direct2.h
#include "transitive1.h"

//--- transitive1.h
#include <print>
void sayGoodbye() {
  std::println("Goodbye!");
}

//--- transitive2.h
// empty
