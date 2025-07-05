// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache \
// RUN:            -fsyntax-only %t/test.c -verify
// Test again with the populated module cache.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache \
// RUN:            -fsyntax-only %t/test.c -verify

// Test that an identifier with the same name as a macro doesn't hide this
// macro from the includers.

//--- macro-definition.h
#define __P(protos) ()
#define __Q(protos) ()

//--- macro-transitive.h
#include "macro-definition.h"
void test(int __P) {} // not "interesting" identifier
struct __Q {};        // "interesting" identifier

//--- module.modulemap
module MacroDefinition { header "macro-definition.h" export * }
module MacroTransitive { header "macro-transitive.h" export * }

//--- test.c
// expected-no-diagnostics
#include "macro-transitive.h"
void foo __P(());
void bar __Q(());
