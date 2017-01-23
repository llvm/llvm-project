// RUN: rm -rf %t.tmp
// RUN: %clang_cc1 -fsyntax-only -I %S/Inputs/SameHeader -fmodules \
// RUN:   -fimplicit-module-maps -fmodules-cache-path=%t.tmp %s -verify

// expected-error@Inputs/SameHeader/C.h:3 {{redefinition of 'c'}}
// expected-note-re@Inputs/SameHeader/B.h:3 {{'{{.*}}/C.h' included multiple times, additional (likely non-modular) include site in module 'X.B'}}
// expected-note-re@Inputs/SameHeader/module.modulemap:6 {{consider adding '{{.*}}/C.h' as part of 'X.B' definition in}}

#include "A.h" // maps to a modular
#include "C.h" // textual include
