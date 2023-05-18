// RUN: rm -rf %t
// RUN: cd %S
//
// RUN: %clang_cc1 -fmodules -x c++ -fmodules-cache-path=%t \
// RUN:   -v \
// RUN:   -iquote Inputs/macro-ambiguity/a/quote \
// RUN:   -isystem Inputs/macro-ambiguity/a/system \
// RUN:   -fno-implicit-modules \
// RUN:   -fmodule-map-file-home-is-cwd \
// RUN:   -emit-module -fmodule-name=a -o %t/a.pcm \
// RUN:   Inputs/macro-ambiguity/module.modulemap
//
// RUN: %clang_cc1 -fmodules -x c++ -fmodules-cache-path=%t \
// RUN:   -v \
// RUN:   -iquote Inputs/macro-ambiguity/b/quote \
// RUN:   -isystem Inputs/macro-ambiguity/b/system \
// RUN:   -fno-implicit-modules \
// RUN:   -fmodule-map-file-home-is-cwd \
// RUN:   -emit-module -fmodule-name=b -o %t/b.pcm \
// RUN:   Inputs/macro-ambiguity/module.modulemap
//
// RUN: %clang_cc1 -fmodules -x c++ -fmodules-cache-path=%t \
// RUN:   -v \
// RUN:   -iquote Inputs/macro-ambiguity/c/quote \
// RUN:   -isystem Inputs/macro-ambiguity/c/system \
// RUN:   -fno-implicit-modules \
// RUN:   -fmodule-map-file-home-is-cwd \
// RUN:   -emit-module -fmodule-name=c -o %t/c.pcm \
// RUN:   Inputs/macro-ambiguity/module.modulemap
//
// RUN: %clang_cc1 -fmodules -x c++ -fmodules-cache-path=%t \
// RUN:   -v \
// RUN:   -iquote Inputs/macro-ambiguity/d/quote \
// RUN:   -isystem Inputs/macro-ambiguity/d/system \
// RUN:   -fno-implicit-modules \
// RUN:   -fmodule-map-file-home-is-cwd \
// RUN:   -emit-module -fmodule-name=d -o %t/d.pcm \
// RUN:   Inputs/macro-ambiguity/module.modulemap
//
// RUN: %clang_cc1 -fmodules -x c++ -fmodules-cache-path=%t \
// RUN:   -v \
// RUN:   -iquote Inputs/macro-ambiguity/a/quote \
// RUN:   -isystem Inputs/macro-ambiguity/a/system \
// RUN:   -iquote Inputs/macro-ambiguity/b/quote \
// RUN:   -isystem Inputs/macro-ambiguity/b/system \
// RUN:   -iquote Inputs/macro-ambiguity/c/quote \
// RUN:   -isystem Inputs/macro-ambiguity/c/system \
// RUN:   -iquote Inputs/macro-ambiguity/d/quote \
// RUN:   -isystem Inputs/macro-ambiguity/d/system \
// RUN:   -iquote Inputs/macro-ambiguity/e/quote \
// RUN:   -isystem Inputs/macro-ambiguity/e/system \
// RUN:   -fno-implicit-modules \
// RUN:   -fmodule-map-file-home-is-cwd \
// RUN:   -fmodule-map-file=Inputs/macro-ambiguity/module.modulemap \
// RUN:   -fmodule-file=%t/a.pcm \
// RUN:   -fmodule-file=%t/b.pcm \
// RUN:   -fmodule-file=%t/c.pcm \
// RUN:   -fmodule-file=%t/d.pcm \
// RUN:   -Wambiguous-macro -verify macro-ambiguity.cpp
//
// RUN: %clang_cc1 -fmodules -x c++ -fmodules-cache-path=%t \
// RUN:   -v -fmodules-local-submodule-visibility \
// RUN:   -iquote Inputs/macro-ambiguity/a/quote \
// RUN:   -isystem Inputs/macro-ambiguity/a/system \
// RUN:   -iquote Inputs/macro-ambiguity/b/quote \
// RUN:   -isystem Inputs/macro-ambiguity/b/system \
// RUN:   -iquote Inputs/macro-ambiguity/c/quote \
// RUN:   -isystem Inputs/macro-ambiguity/c/system \
// RUN:   -iquote Inputs/macro-ambiguity/d/quote \
// RUN:   -isystem Inputs/macro-ambiguity/d/system \
// RUN:   -iquote Inputs/macro-ambiguity/e/quote \
// RUN:   -isystem Inputs/macro-ambiguity/e/system \
// RUN:   -fno-implicit-modules \
// RUN:   -fmodule-map-file-home-is-cwd \
// RUN:   -fmodule-map-file=Inputs/macro-ambiguity/module.modulemap \
// RUN:   -fmodule-file=%t/a.pcm \
// RUN:   -fmodule-file=%t/b.pcm \
// RUN:   -fmodule-file=%t/c.pcm \
// RUN:   -fmodule-file=%t/d.pcm \
// RUN:   -Wambiguous-macro -verify macro-ambiguity.cpp

// Include the textual headers first to maximize the ways in which things can
// become ambiguous.
#include "e_quote.h"
#include <e_system.h>

#include "a_quote.h"
#include <a_system.h>
#include "b_quote.h"
#include <b_system.h>
#include "c_quote.h"
#include <c_system.h>
#include "d_quote.h"
#include <d_system.h>

int test(int x) {
  // We expect to get warnings for all of the quoted includes but none of the
  // system includes here because the first module is a non-system module and
  // the quote macros come from non-system-headers.
  x = FOO1_QUOTE(x); // expected-warning {{ambiguous expansion of macro}}
  // expected-note@Inputs/macro-ambiguity/c/quote/c_quote.h:4 {{expanding this definition}}
  // expected-note@Inputs/macro-ambiguity/a/quote/a_quote.h:4 {{other definition}}

  x = FOO1_SYSTEM(x);

  x = BAR1_QUOTE(x); // expected-warning {{ambiguous expansion of macro}}
  // expected-note@Inputs/macro-ambiguity/d/quote/d_quote.h:4 {{expanding this definition}}
  // expected-note@Inputs/macro-ambiguity/a/quote/a_quote.h:5 {{other definition}}

  x = BAR1_SYSTEM(x);

  x = BAZ1_QUOTE(x); // expected-warning {{ambiguous expansion of macro}}
  // expected-note@Inputs/macro-ambiguity/a/quote/a_quote.h:6 {{expanding this definition}}
  // expected-note@Inputs/macro-ambiguity/e/quote/e_quote.h:4 {{other definition}}

  x = BAZ1_SYSTEM(x);

  // Here, we don't even warn on bar because in that cas both b and d are
  // system modules and so the use of non-system headers is irrelevant.
  x = FOO2_QUOTE(x); // expected-warning {{ambiguous expansion of macro}}
  // expected-note@Inputs/macro-ambiguity/c/quote/c_quote.h:5 {{expanding this definition}}
  // expected-note@Inputs/macro-ambiguity/b/quote/b_quote.h:4 {{other definition}}

  x = FOO2_SYSTEM(x);

  x = BAR2_QUOTE(x);

  x = BAR2_SYSTEM(x);

  x = BAZ2_QUOTE(x); // expected-warning {{ambiguous expansion of macro}}
  // expected-note@Inputs/macro-ambiguity/b/quote/b_quote.h:6 {{expanding this definition}}
  // expected-note@Inputs/macro-ambiguity/e/quote/e_quote.h:5 {{other definition}}

  x = BAZ2_SYSTEM(x);
  return x;
}
