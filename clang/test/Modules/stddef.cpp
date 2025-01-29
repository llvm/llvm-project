// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/no-lsv -I%t %t/stddef.cpp -verify
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-local-submodule-visibility -fmodules-cache-path=%t/lsv -I%t %t/stddef.cpp -verify

//--- stddef.cpp
#include <b.h>

void *pointer = NULL;
size_t size = 0;

// When building with modules, a pcm is never re-imported, so re-including
// stddef.h will not re-import _Builtin_stddef.null to restore the definition of
// NULL, even though stddef.h will unconditionally include __stddef_null.h when
// building with modules.
#undef NULL
#include <stddef.h>

void *anotherPointer = NULL; // expected-error{{use of undeclared identifier 'NULL'}}

// stddef.h needs to be a `textual` header to support clients doing things like
// this.
//
// #define __need_NULL
// #include <stddef.h>
//
// As a textual header designed to be included multiple times, it can't directly
// declare anything, or those declarations would go into every module that
// included it. e.g. if stddef.h contained all of its declarations, and modules
// A and B included stddef.h, they would both have the declaration for size_t.
// That breaks Swift, which uses the module name as part of the type name, i.e.
// A.size_t and B.size_t are treated as completely different types in Swift and
// cannot be interchanged. To fix that, stddef.h (and stdarg.h) are split out
// into a separate file per __need macro that can be normal headers in explicit
// submodules. That runs into yet another wrinkle though. When modules build,
// declarations from previous submodules leak into subsequent ones when not
// using local submodule visibility. Consider if stddef.h did the normal thing.
//
// #ifndef __STDDEF_H
// #define __STDDEF_H
// // include all of the sub-headers
// #endif
//
// When SM builds without local submodule visibility, it will precompile a.h
// first. When it gets to b.h, the __STDDEF_H declaration from precompiling a.h
// will leak, and so when b.h includes stddef.h, it won't include any of its
// sub-headers, and SM.B will thus not import _Builtin_stddef or make any of its
// submodules visible. Precompiling b.h will be fine since it sees all of the
// declarations from a.h including stddef.h, but clients that only include b.h
// will not see any of the stddef.h types. stddef.h thus has to make sure to
// always include the necessary sub-headers, even if they've been included
// already. They all have their own header guards to allow this.
// __stddef_null.h is extra special, so this test makes sure to cover NULL plus
// one of the normal stddef.h types.

//--- module.modulemap
module SM {
  module A {
    header "a.h"
    export *
  }

  module B {
    header "b.h"
    export *
  }
}

//--- a.h
#include <stddef.h>

//--- b.h
#include <stddef.h>
