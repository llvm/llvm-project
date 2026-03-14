// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/a.cppm -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/b.cppm -o %t/b.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/c.cppm -o %t/c.pcm
//
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t -emit-module-interface %t/aggregate.internal.cppm -o %t/aggregate.internal.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t -emit-module-interface %t/aggregate.cppm -o %t/aggregate.pcm
//
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/use.cpp -verify -DTEST


//--- a.cppm
export module a;
export class A{};

//--- b.cppm
export module b;
export class B{};

//--- c.cppm
export module c;
export class C{};

//--- aggregate.internal.cppm
export module aggregate.internal;
export import a;
export import b;
export import c;

//--- aggregate.cppm
// Export the above aggregate module.
// This is done to ensure that re-exports are transitive.
export module aggregate;
export import aggregate.internal;


//--- use.cpp
// expected-no-diagnostics
// For the actual test, just try using the classes from the exported modules
// and hope that they're accessible.
import aggregate;
A a;
B b;
C c;
