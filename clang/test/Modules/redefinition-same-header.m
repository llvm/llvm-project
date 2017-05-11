// RUN: rm -rf %t.tmp
// RUN: %clang_cc1 -fsyntax-only -I %S/Inputs/SameHeader -fmodules \
// RUN:   -fimplicit-module-maps -fmodules-cache-path=%t.tmp %s -verify

// expected-error@Inputs/SameHeader/C.h:3 {{redefinition of 'c'}}
// expected-note-re@Inputs/SameHeader/B.h:3 {{'{{.*}}C.h' included multiple times, additional include site in header from module 'X.B'}}
// expected-note@Inputs/SameHeader/module.modulemap:6 {{X.B defined here}}
// expected-note-re@redefinition-same-header.m:20 {{'{{.*}}C.h' included multiple times, additional include site here}}

// expected-error@Inputs/SameHeader/C.h:5 {{redefinition of 'aaa'}}
// expected-note-re@Inputs/SameHeader/B.h:3 {{'{{.*}}C.h' included multiple times, additional include site in header from module 'X.B'}}
// expected-note@Inputs/SameHeader/module.modulemap:6 {{X.B defined here}}
// expected-note-re@redefinition-same-header.m:20 {{'{{.*}}C.h' included multiple times, additional include site here}}

// expected-error@Inputs/SameHeader/C.h:9 {{redefinition of 'fd_set'}}
// expected-note-re@Inputs/SameHeader/B.h:3 {{'{{.*}}C.h' included multiple times, additional include site in header from module 'X.B'}}
// expected-note@Inputs/SameHeader/module.modulemap:6 {{X.B defined here}}
// expected-note-re@redefinition-same-header.m:20 {{'{{.*}}C.h' included multiple times, additional include site here}}
#include "A.h" // maps to a modular
#include "C.h" // textual include
