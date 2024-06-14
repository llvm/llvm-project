// RUN: split-file --leading-lines %s %t
// RUN: %clang_cc1 -std=c++20 -pedantic-errors -verify -emit-module-interface %t/module.cppm -o %t/module.pcm
// RUN: %clang_cc1 -std=c++20 -pedantic-errors -verify -fmodule-file=A=%t/module.pcm %t/main.cpp
// RUN: %clang_cc1 -std=c++23 -pedantic-errors -verify -emit-module-interface %t/module.cppm -o %t/module.pcm
// RUN: %clang_cc1 -std=c++23 -pedantic-errors -verify -fmodule-file=A=%t/module.pcm %t/main.cpp
// RUN: %clang_cc1 -std=c++2c -pedantic-errors -verify -emit-module-interface %t/module.cppm -o %t/module.pcm
// RUN: %clang_cc1 -std=c++2c -pedantic-errors -verify -fmodule-file=A=%t/module.pcm %t/main.cpp

//--- module.cppm
// expected-no-diagnostics
export module A;

namespace cwg2630 {
export class X {};
} // namespace cwg2630

//--- main.cpp
// expected-no-diagnostics
import A;

namespace cwg2630 { // cwg2630: 9
X x;
} // namespace cwg2630
