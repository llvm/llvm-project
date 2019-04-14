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
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.y.pcm -verify %s \
// RUN:            -DTEST=1 -DEXPORT= -DMODULE_NAME=z
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.y.pcm -verify %s \
// RUN:            -DTEST=2 -DEXPORT= -DMODULE_NAME=x
//
// Module interface for unknown and known module. (The latter is ill-formed due to
// redefinition.)
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.y.pcm -verify %s \
// RUN:            -DTEST=3 -DEXPORT=export -DMODULE_NAME=z
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.y.pcm -verify %s \
// RUN:            -DTEST=4 -DEXPORT=export -DMODULE_NAME=x
//
// Miscellaneous syntax.
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.y.pcm -verify %s \
// RUN:            -DTEST=7 -DEXPORT=export -DMODULE_NAME='z elderberry'
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.y.pcm -verify %s \
// RUN:            -DTEST=8 -DEXPORT=export -DMODULE_NAME='z [[]]'
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.y.pcm -verify %s \
// RUN:            -DTEST=9 -DEXPORT=export -DMODULE_NAME='z [[fancy]]'
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.y.pcm -verify %s \
// RUN:            -DTEST=10 -DEXPORT=export -DMODULE_NAME='z [[maybe_unused]]'

EXPORT module MODULE_NAME;
#if TEST == 4
// expected-error@-2 {{redefinition of module 'x'}}
// expected-note-re@module-declaration.cpp:* {{loaded from '{{.*[/\\]}}x.pcm'}}
#elif TEST == 7
// expected-error@-5 {{expected ';'}} expected-error@-5 {{requires a type specifier}}
#elif TEST == 9
// expected-warning@-7 {{unknown attribute 'fancy' ignored}}
#elif TEST == 10
// expected-error-re@-9 {{'maybe_unused' attribute cannot be applied to a module{{$}}}}
#elif TEST == 1
// expected-error@-11 {{definition of module 'z' is not available}}
#else
// expected-no-diagnostics
#endif
