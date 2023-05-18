// Based on C++20 10.2 example 2.

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -emit-header-unit -I %t \
// RUN: -xc++-user-header std-10-2-ex2-c.h -o %t/std-10-2-ex2-c.pcm

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/std-10-2-ex2-tu1.cpp \
// RUN:  -o  %t/X.pcm

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/std-10-2-ex2-tu2.cpp \
// RUN: -fmodule-file=%t/std-10-2-ex2-c.pcm -fmodule-file=X=%t/X.pcm \
// RUN: -pedantic-errors -verify -o  %t/M.pcm

//--- std-10-2-ex2-b.h
int f();

//--- std-10-2-ex2-c.h
int g();

//--- std-10-2-ex2-tu1.cpp
export module X;
export int h();

//--- std-10-2-ex2-tu2.cpp
module;
#include "std-10-2-ex2-b.h"

export module M;
import "std-10-2-ex2-c.h";  // expected-warning {{the implementation of header units is in an experimental phase}}
import X;
export using ::f, ::g, ::h; // OK
struct S;                   // expected-note {{target of using declaration}}
export using ::S;           // expected-error {{using declaration referring to 'S' with module linkage cannot be exported}}

namespace N {
export int h();
static int h(int); // expected-note {{target of using declaration}}
} // namespace N
export using N::h; // expected-error {{using declaration referring to 'h' with internal linkage cannot be exported}}
