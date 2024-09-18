// Test -D and -U interaction with a PCH when -fms-extensions is enabled.

// RUN: %clang_cc1 -DFOO %S/variables.h -emit-pch -o %t1.pch

// RUN: not %clang_cc1 -DFOO=blah -DBAR=int -include-pch %t1.pch -pch-through-header=%S/variables.h %s 2> %t.err
// RUN: FileCheck -check-prefix=CHECK-FOO %s < %t.err

// RUN: not %clang_cc1 -UFOO -DBAR=int -include-pch %t1.pch %s -pch-through-header=%S/variables.h 2> %t.err
// RUN: FileCheck -check-prefix=CHECK-NOFOO %s < %t.err

// RUN: %clang_cc1 -include-pch %t1.pch -DBAR=int -pch-through-header=%S/variables.h -verify %s

// Enabling MS extensions should allow us to add BAR definitions.
// RUN: %clang_cc1 -DMSEXT -fms-extensions -DFOO %S/variables.h -emit-pch -o %t1.pch
// RUN: %clang_cc1 -DMSEXT -fms-extensions -include-pch %t1.pch -DBAR=int -pch-through-header=%S/variables.h -verify %s

#include "variables.h"

BAR bar = 17;
#ifndef MSEXT
// expected-error@-2 {{unknown type name 'BAR'}}
#endif

#ifndef FOO
#  error FOO was not defined
#endif

#if FOO != 1
#  error FOO has the wrong definition
#endif

#if defined(MSEXT) && !defined(BAR)
#  error BAR was not defined
#endif

// CHECK-FOO: definition of macro 'FOO' differs between the AST file '{{.*}}1.pch' ('1') and the command line ('blah')
// CHECK-NOFOO: macro 'FOO' was defined in the AST file '{{.*}}1.pch' but undef'd on the command line

// expected-warning@2 {{definition of macro 'BAR' does not match definition in precompiled header}}
