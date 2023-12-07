// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 -xc++-user-header h1.h -emit-header-unit -o h1.pcm
// RUN: %clang_cc1 -std=c++20 -xc++-user-header h2.h -emit-header-unit -o h2.pcm
// RUN: %clang_cc1 -std=c++20 -xc++-user-header h3.h -emit-header-unit -o h3.pcm
// RUN: %clang_cc1 -std=c++20 -xc++-user-header h4.h -emit-header-unit -o h4.pcm

// RUN: %clang_cc1 -std=c++20 Xlate.cpp -emit-module-interface -o Xlate.pcm \
// RUN: -fmodule-file=h1.pcm -fmodule-file=h2.pcm -fmodule-file=h3.pcm \
// RUN: -fmodule-file=h4.pcm -fsyntax-only -Rmodule-include-translation -verify

// Check that we do the intended translation and not more.
// RUN: %clang_cc1 -std=c++20 Xlate.cpp \
// RUN: -fmodule-file=h1.pcm -fmodule-file=h2.pcm -fmodule-file=h3.pcm \
// RUN: -fmodule-file=h4.pcm  -E -undef | FileCheck %s

// We expect no diagnostics here, the used functions should all be available.
// RUN: %clang_cc1 -std=c++20 Xlate.cpp -emit-module-interface \
// RUN: -fmodule-file=h1.pcm -fmodule-file=h2.pcm -fmodule-file=h3.pcm \
// RUN: -fmodule-file=h4.pcm -fsyntax-only

// The content of the headers is not terribly important, we just want to check
// whether they are textually included or include-translated.
//--- h1.h
#ifndef H1_GUARD
#define H1_GUARD

#define ONE 1

void foo();

#endif // H1_GUARD

//--- h2.h
#ifndef H2_GUARD
#define H2_GUARD

#define TWO 2

void bar();

#endif // H2_GUARD

//--- h3.h
#ifndef H3_GUARD
#define H3_GUARD

#define THREE 3

void baz();

#endif // H3_GUARD

//--- h4.h
#ifndef H4_GUARD
#define H4_GUARD

#define FOUR 4

void boo();

#endif // H4_GUARD

//--- h5.h
#ifndef H5_GUARD
#define H5_GUARD

#define FIVE 5

void five();

#endif // H4_GUARD

//--- Xlate.cpp
/* some comment ...
  ... */
module /*nothing here*/;

// This should be include-translated, when the header unit for h1 is available.
 // expected-warning@+1 {{the implementation of header units is in an experimental phase}}
#include "h1.h" // expected-remark {{treating #include as an import of module './h1.h'}}
// Import of a header unit is allowed, named modules are not.
import "h2.h"; // expected-warning {{the implementation of header units is in an experimental phase}}
// A regular, untranslated, header
#include "h5.h"

export module Xlate;

// This is OK, the import immediately follows the module decl.
import "h3.h"; // expected-warning {{the implementation of header units is in an experimental phase}}

// This should *not* be include-translated, even if header unit for h4 is
// available.
#include "h4.h"

export void charlie() {
  foo();
  bar();
  baz();
  boo();
  five();
}

// CHECK: #pragma clang module import "./h1.h"
// CHECK: import ./h2.h
// CHECK: import ./h3.h
// CHECK-NOT: #pragma clang module import "./h4.h"
