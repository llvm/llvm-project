// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t

// RUN: %clang_cc1 -std=c++20 A-intf-part.cpp -emit-module-interface \
// RUN:  -o A-PubPart.pcm
// RUN: %clang_cc1 -std=c++20 A-interface.cpp -emit-module-interface \
// RUN:   -fmodule-file=A-PubPart.pcm -o A.pcm

// RUN: %clang_cc1 -std=c++20 A-impl-top.cpp -fsyntax-only -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 A-impl-part.cpp -fsyntax-only -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 A-impl-1.cpp -fsyntax-only -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 A-impl-2.cpp -fsyntax-only -fprebuilt-module-path=%t

//--- A-interface.cpp
export module A;

export import :PubPart;

export void do_something();

void helper1();
void helper3();

//--- A-intf-part.cpp
export module A:PubPart;

void helper2();

//--- A-impl-top.cpp

module A;

void do_something() {
  helper1();
  helper2();
  helper3();
}

//--- A-impl-part.cpp
module A:Secret;

import A;

void helper3() {}

//--- A-impl-1.cpp
module A;

void helper1() {}

//--- A-impl-2.cpp
module A;

void helper2() {}
