
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/std10-1-ex2-tu1.cpp \
// RUN:  -o %t/B_Y.pcm

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/std10-1-ex2-tu2.cpp \
// RUN:  -fmodule-file=%t/B_Y.pcm -o %t/B.pcm

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/std10-1-ex2-tu3.cpp \
// RUN:   -o %t/B_X1.pcm -verify

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/std10-1-ex2-tu4.cpp \
// RUN:-fmodule-file=%t/B.pcm  -o %t/B_X2.pcm

// RUN: %clang_cc1 -std=c++20 -emit-obj %t/std10-1-ex2-tu5.cpp \
// RUN:  -fmodule-file=%t/B.pcm  -o %t/b_tu5.o

// RUN: %clang_cc1 -std=c++20 -S %t/std10-1-ex2-tu6.cpp \
// RUN:  -fmodule-file=%t/B.pcm  -o %t/b_tu6.s -verify

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/std10-1-ex2-tu7.cpp \
// RUN: -fmodule-file=%t/B_X2.pcm  -o %t/B_X3.pcm -verify

//--- std10-1-ex2-tu1.cpp
module B:Y;
int y();
// expected-no-diagnostics

//--- std10-1-ex2-tu2.cpp
export module B;
import :Y;
int n = y();
// expected-no-diagnostics

//--- std10-1-ex2-tu3.cpp
module B:X1; // does not implicitly import B
int &a = n;  // expected-error {{use of undeclared identifier }}

//--- std10-1-ex2-tu4.cpp
module B:X2; // does not implicitly import B
import B;
int &b = n; // OK
// expected-no-diagnostics

//--- std10-1-ex2-tu5.cpp
module B;   // implicitly imports B
int &c = n; // OK
// expected-no-diagnostics

//--- std10-1-ex2-tu6.cpp
import B;
// error, n is module-local and this is not a module.
int &c = n; // expected-error {{use of undeclared identifier}}

//--- std10-1-ex2-tu7.cpp
module B:X3; // does not implicitly import B
import :X2;  // X2 is an implementation so exports nothing.
             // error: n not visible here.
int &c = n;  // expected-error {{use of undeclared identifier }}
