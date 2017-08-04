// Tests for module-declaration syntax.
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: echo 'export module x; int a, b;' > %t/x.cppm
// RUN: echo 'export module x.y; int c;' > %t/x.y.cppm
//
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -emit-module-interface %t/x.cppm -o %t/x.pcm
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -emit-module-interface -fmodule-file=%t/x.pcm %t/x.y.cppm -o %t/x.y.pcm
//
// Module implementation for unknown and known module. (The former is ill-formed.)
// FIXME: TEST=1 should fail because we don't have an interface for module z.
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.y.pcm -verify %s \
// RUN:            -DTEST=1 -DEXPORT= -DPARTITION= -DMODULE_NAME=z
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.y.pcm -verify %s \
// RUN:            -DTEST=2 -DEXPORT= -DPARTITION= -DMODULE_NAME=x
//
// Module interface for unknown and known module. (The latter is ill-formed due to
// redefinition.)
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.y.pcm -verify %s \
// RUN:            -DTEST=3 -DEXPORT=export -DPARTITION= -DMODULE_NAME=z
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.y.pcm -verify %s \
// RUN:            -DTEST=4 -DEXPORT=export -DPARTITION= -DMODULE_NAME=x
//
// Defining a module partition.
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.y.pcm -verify %s \
// RUN:            -DTEST=5 -DEXPORT=export -DPARTITION=partition -DMODULE_NAME=z
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.y.pcm -verify %s \
// RUN:            -DTEST=6 -DEXPORT= -DPARTITION=partition -DMODULE_NAME=z
//
// Miscellaneous syntax.
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.y.pcm -verify %s \
// RUN:            -DTEST=7 -DEXPORT= -DPARTITION=elderberry -DMODULE_NAME=z
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.y.pcm -verify %s \
// RUN:            -DTEST=8 -DEXPORT= -DPARTITION= -DMODULE_NAME='z [[]]'
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.y.pcm -verify %s \
// RUN:            -DTEST=9 -DEXPORT= -DPARTITION= -DMODULE_NAME='z [[fancy]]'
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.y.pcm -verify %s \
// RUN:            -DTEST=10 -DEXPORT= -DPARTITION= -DMODULE_NAME='z [[maybe_unused]]'

EXPORT module PARTITION MODULE_NAME;
#if TEST == 4
// expected-error@-2 {{redefinition of module 'x'}}
// expected-note-re@module-declaration.cpp:* {{loaded from '{{.*}}/x.pcm'}}
#elif TEST == 6
// expected-error@-5 {{module partition must be declared 'export'}}
#elif TEST == 7
// expected-error@-7 {{expected ';'}} expected-error@-7 {{requires a type specifier}}
#elif TEST == 9
// expected-warning@-9 {{unknown attribute 'fancy' ignored}}
#elif TEST == 10
// expected-error-re@-11 {{'maybe_unused' attribute cannot be applied to a module{{$}}}}
#else
// expected-no-diagnostics
#endif
