// This test checks that module cache populated using different spellings of the
// same underlying directory works consistently (specifically the IMPORT records.)

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: mkdir %t/tmp

// Module cache path absolute.
// RUN: cd %t/tmp && %clang_cc1 -fmodules -fimplicit-module-maps -fsyntax-only -I %t %t/tu1.c -Rmodule-build -Rmodule-import -verify -fmodules-cache-path=%t/cache
//--- tu1.c
#include "b.h" // expected-remark{{building module 'b'}}          \
               // expected-remark{{finished building module 'b'}} \
               // expected-remark{{importing module 'b'}}         \
               // expected-remark{{importing module 'a' into 'b'}}

// Module cache path relative to CWD.
// RUN: cd %t && %clang_cc1 -fmodules -fimplicit-module-maps -fsyntax-only -I %t %t/tu2.c -Rmodule-build -Rmodule-import -verify -fmodules-cache-path=cache
//--- tu2.c
#include "c.h" // expected-remark{{building module 'c'}}           \
               // expected-remark{{finished building module 'c'}}  \
               // expected-remark{{importing module 'c'}}          \
               // expected-remark{{importing module 'b' into 'c'}} \
               // expected-remark{{importing module 'a' into 'b'}}

// Module cache path relative to -working-directory.
// RUN: cd %t/tmp && %clang_cc1 -fmodules -fimplicit-module-maps -fsyntax-only -I %t %t/tu3.c -Rmodule-build -Rmodule-import -verify -fmodules-cache-path=cache -working-directory %t
//--- tu3.c
#include "d.h" // expected-remark{{building module 'd'}}           \
               // expected-remark{{finished building module 'd'}}  \
               // expected-remark{{importing module 'd'}}          \
               // expected-remark{{importing module 'c' into 'd'}} \
               // expected-remark{{importing module 'b' into 'c'}} \
               // expected-remark{{importing module 'a' into 'b'}}

// Module cache path absolute again.
// RUN: cd %t/tmp && %clang_cc1 -fmodules -fimplicit-module-maps -fsyntax-only -I %t %t/tu4.c -Rmodule-build -Rmodule-import -verify -fmodules-cache-path=%t/cache
//--- tu4.c
#include "e.h" // expected-remark{{building module 'e'}}           \
               // expected-remark{{finished building module 'e'}}  \
               // expected-remark{{importing module 'e'}}          \
               // expected-remark{{importing module 'd' into 'e'}} \
               // expected-remark{{importing module 'c' into 'd'}} \
               // expected-remark{{importing module 'b' into 'c'}} \
               // expected-remark{{importing module 'a' into 'b'}}

//--- module.modulemap
module a { header "a.h" }
module b { header "b.h" }
module c { header "c.h" }
module d { header "d.h" }
module e { header "e.h" }
//--- a.h
//--- b.h
#include "a.h"
//--- c.h
#include "b.h"
//--- d.h
#include "c.h"
//--- e.h
#include "d.h"
