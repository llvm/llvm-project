// RUN: rm -rf %t
// RUN: mkdir -p %t

// RUN: %clang_cc1 -std=c++20 -emit-module-interface -D TU=0 -x c++ %s \
// RUN:  -o %t/B.pcm

// RUN: %clang_cc1 -std=c++20 -emit-module-interface -D TU=1 -x c++ %s \
// RUN:  -o %t/C.pcm

// RUN: %clang_cc1 -std=c++20 -emit-module-interface -D TU=2 -x c++ %s \
// RUN:  -fmodule-file=%t/B.pcm -fmodule-file=%t/C.pcm -o %t/AOK1.pcm

// RUN: %clang_cc1 -std=c++20 -S -D TU=3 -x c++ %s \
// RUN:  -fmodule-file=%t/AOK1.pcm -o %t/tu_3.s -verify

// RUN: %clang_cc1 -std=c++20 -emit-module-interface -D TU=4 -x c++ %s \
// RUN:  -fmodule-file=%t/B.pcm -fmodule-file=%t/C.pcm -o %t/BC.pcm -verify

// RUN: %clang_cc1 -std=c++20 -S -D TU=5 -x c++ %s \
// RUN:  -fmodule-file=%t/B.pcm -fmodule-file=%t/C.pcm -o %t/tu_5.s -verify

// RUN: %clang_cc1 -std=c++20 -emit-module-interface -D TU=6 -x c++ %s \
// RUN:  -fmodule-file=%t/B.pcm -o %t/D.pcm -verify

// RUN: %clang_cc1 -std=c++20 -emit-module-interface -D TU=7 -x c++ %s \
// RUN:  -fmodule-file=%t/B.pcm -o %t/D.pcm -verify

// RUN: %clang_cc1 -std=c++20 -S -D TU=8 -x c++ %s \
// RUN:  -fmodule-file=%t/B.pcm -o %t/tu_8.s -verify

// RUN: %clang_cc1 -std=c++20 -emit-module-interface -D TU=9 -x c++ %s \
// RUN:  -o %t/B.pcm -verify

// RUN: %clang_cc1 -std=c++20 -emit-obj -D TU=10 -x c++ %s \
// RUN:  -fmodule-file=%t/C.pcm  -o %t/impl.o

// Test diagnostics for incorrect module import sequences.

#if TU == 0

export module B;

int foo ();

// expected-no-diagnostics

#elif TU == 1

export module C;

int bar ();

// expected-no-diagnostics

#elif TU == 2

export module AOK1;

import B;
export import C;

export int theAnswer ();

// expected-no-diagnostics

#elif TU == 3

module;

module AOK1;

export import C; // expected-error {{export declaration can only be used within a module interface unit}}

int theAnswer () { return 42; }

#elif TU == 4

export module BC;

export import B;

int foo () { return 10; }

import C; // expected-error {{imports must immediately follow the module declaration}}

#elif TU == 5

module B; // implicitly imports B.

int foo () { return 10; }

import C; // expected-error {{imports must immediately follow the module declaration}}

#elif TU == 6

module;
// We can only have preprocessor commands here, which could include an include
// translated header unit.  However those are identified specifically by the
// preprocessor; non-preprocessed user code should not contain an import here.
import B; // expected-error {{module imports cannot be in the global module fragment}}

export module D;

int delta ();

#elif TU == 7

export module D;

int delta ();

module :private;

import B; // expected-error {{module imports cannot be in the private module fragment}}

#elif TU == 8

module B;

import B; // expected-error {{import of module 'B' appears within same top-level module 'B'}}

#elif TU == 9

export module B;

import B; // expected-error {{import of module 'B' appears within same top-level module 'B'}}

#elif TU == 10

int x;

import C;

int baz() { return 6174; }

// expected-no-diagnostics

#else
#error "no MODE set"
#endif
