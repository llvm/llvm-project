
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/std10-1-ex2-tu1.cpp \
// RUN:  -o %t/B_Y.pcm
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/std10-1-ex2-tu2.cpp \
// RUN:     -fmodule-file=B:Y=%t/B_Y.pcm -o %t/B.pcm
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/std10-1-ex2-tu3.cpp \
// RUN:     -o %t/B_X1.pcm -verify
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/std10-1-ex2-tu4.cpp \
// RUN:     -fmodule-file=B=%t/B.pcm -fmodule-file=B:Y=%t/B_Y.pcm  -o %t/B_X2.pcm
//
// RUN: %clang_cc1 -std=c++20 -emit-obj %t/std10-1-ex2-tu5.cpp \
// RUN:     -fmodule-file=B=%t/B.pcm -fmodule-file=B:Y=%t/B_Y.pcm  -o %t/b_tu5.o
//
// RUN: %clang_cc1 -std=c++20 -S %t/std10-1-ex2-tu6.cpp \
// RUN:     -fmodule-file=B=%t/B.pcm -fmodule-file=B:Y=%t/B_Y.pcm  -o %t/b_tu6.s -verify
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/std10-1-ex2-tu7.cpp \
// RUN:     -fmodule-file=B:X2=%t/B_X2.pcm -fmodule-file=B=%t/B.pcm \
// RUN:     -fmodule-file=B:Y=%t/B_Y.pcm   -o %t/B_X3.pcm -verify

// Test again with reduced BMI.
// RUN: rm %t/B_X2.pcm %t/B.pcm %t/B_Y.pcm
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/std10-1-ex2-tu1.cpp \
// RUN:  -o %t/B_Y.pcm
//
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/std10-1-ex2-tu2.cpp \
// RUN:     -fmodule-file=B:Y=%t/B_Y.pcm -o %t/B.pcm
//
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/std10-1-ex2-tu3.cpp \
// RUN:     -o %t/B_X1.pcm -verify
//
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/std10-1-ex2-tu4.cpp \
// RUN:     -fmodule-file=B=%t/B.pcm -fmodule-file=B:Y=%t/B_Y.pcm  -o %t/B_X2.pcm
//
// RUN: %clang_cc1 -std=c++20 -emit-obj %t/std10-1-ex2-tu5.cpp \
// RUN:     -fmodule-file=B=%t/B.pcm -fmodule-file=B:Y=%t/B_Y.pcm  -o %t/b_tu5.o
//
// RUN: %clang_cc1 -std=c++20 -S %t/std10-1-ex2-tu6.cpp \
// RUN:     -fmodule-file=B=%t/B.pcm -fmodule-file=B:Y=%t/B_Y.pcm  -o %t/b_tu6.s -verify
//
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/std10-1-ex2-tu7.cpp \
// RUN:     -fmodule-file=B:X2=%t/B_X2.pcm -fmodule-file=B=%t/B.pcm \
// RUN:     -fmodule-file=B:Y=%t/B_Y.pcm   -o %t/B_X3.pcm -verify

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
int &c = n; // expected-error {{declaration of 'n' must be imported}}
            // expected-note@* {{declaration here is not visible}}

//--- std10-1-ex2-tu7.cpp
// expected-no-diagnostics
module B:X3; // does not implicitly import B
import :X2; // X2 is an implementation unit import B.
// According to [module.import]p7:
//   Additionally, when a module-import-declaration in a module unit of some
//   module M imports another module unit U of M, it also imports all
//   translation units imported by non-exported module-import-declarations in
//   the module unit purview of U.
//
// So B is imported in B:X3 due to B:X2 imported B. So n is visible here.
int &c = n;
