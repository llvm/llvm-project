// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t

// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-header-unit-header std-10-6-ex1-decl.h \
// RUN: -o decl.pcm

// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-header-unit-header std-10-6-ex1-defn.h \
// RUN: -o defn.pcm

// RUN: %clang_cc1 -std=c++20 -emit-module-interface std-10-6-ex1-stuff.cpp \
// RUN: -o stuff.pcm

// RUN: %clang_cc1 -std=c++20 -emit-module-interface std-10-6-ex1-M1.cpp \
// RUN:   -fmodule-file=stuff=stuff.pcm -o M1.pcm  -fmodule-file=defn.pcm

// RUN: %clang_cc1 -std=c++20 -emit-module-interface std-10-6-ex1-M2.cpp \
// RUN:   -fmodule-file=stuff=stuff.pcm -o M2.pcm  -fmodule-file=decl.pcm

// RUN: %clang_cc1 -std=c++20 std-10-6-ex1-use.cpp \
// RUN:   -fmodule-file=M1=M1.pcm -fmodule-file=M2=M2.pcm -fmodule-file=stuff=stuff.pcm \
// RUN:   -fsyntax-only -verify

//--- std-10-6-ex1-decl.h
struct X;

//--- std-10-6-ex1-defn.h
struct X {};

//--- std-10-6-ex1-stuff.cpp
export module stuff;
export template <typename T, typename U> void foo(T, U u) { auto v = u; }
export template <typename T, typename U> void bar(T, U u) { auto v = *u; }

//--- std-10-6-ex1-M1.cpp
export module M1;
import "std-10-6-ex1-defn.h"; // provides struct X {};
import stuff;

export template <typename T> void f(T t) {
  X x;
  foo(t, x);
}

//--- std-10-6-ex1-M2.cpp
export module M2;
import "std-10-6-ex1-decl.h"; // provides struct X; (not a definition)

import stuff;
export template <typename T> void g(T t) {
  X *x;
  bar(t, x);
}

//--- std-10-6-ex1-use.cpp
import M1;
import M2;

void test() {
  f(0);
  // It is unspecified whether the instantiation of g(0) is valid here.
  // We choose to make it invalid here.
  g(0); // expected-error@* {{definition of 'X' must be imported from module}}
        // expected-note@* {{in instantiation of function template specialization 'bar<int, X *>'}}
        // expected-note@* {{in instantiation of function template specialization}}
        // expected-note@* {{definition here is not reachable}}
}
