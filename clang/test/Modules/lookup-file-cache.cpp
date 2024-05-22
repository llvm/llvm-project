// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -I %t/include -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/cache -fsyntax-only %t/tu.c

//--- include/module.modulemap
module A [no_undeclared_includes] { textual header "A.h" }
module B { header "B.h" }

//--- include/A.h
#if __has_include(<B.h>)
#error B.h should not be available from A.h.
#endif

//--- include/B.h
// This file intentionally left blank.

//--- tu.c
#if !__has_include(<B.h>)
#error B.h should be available from tu.c.
#endif

#include "A.h"

#if !__has_include(<B.h>)
#error B.h should still be available from tu.c.
#endif
