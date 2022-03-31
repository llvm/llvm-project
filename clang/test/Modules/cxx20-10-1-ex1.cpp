// The example in the standard is not in required build order.
// revised here

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/std10-1-ex1-tu1.cpp \
// RUN:  -o %t/A_Internals.pcm

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/std10-1-ex1-tu2.cpp \
// RUN:  -fmodule-file=%t/A_Internals.pcm -o %t/A_Foo.pcm

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/std10-1-ex1-tu3.cpp \
// RUN:  -fmodule-file=%t/A_Foo.pcm -o %t/A.pcm

// RUN: %clang_cc1 -std=c++20 -emit-obj %t/std10-1-ex1-tu4.cpp \
// RUN:  -fmodule-file=%t/A.pcm -o %t/ex1.o

// expected-no-diagnostics

//--- std10-1-ex1-tu1.cpp

module A:Internals;
int bar();

//--- std10-1-ex1-tu2.cpp

export module A:Foo;

import :Internals;

export int foo() { return 2 * (bar() + 1); }

//--- std10-1-ex1-tu3.cpp

export module A;
export import :Foo;
export int baz();

//--- std10-1-ex1-tu4.cpp

module A;

import :Internals;

int bar() { return baz() - 10; }
int baz() { return 30; }
