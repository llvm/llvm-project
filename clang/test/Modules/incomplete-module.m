@import incomplete_mod;

// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -Wincomplete-module -fmodules -fimplicit-module-maps -I %S/Inputs %s 2>&1 | FileCheck %s
// CHECK: warning: include of non-modular header inside module 'incomplete_mod'

// RUN: rm -rf %t
// RUN: not %clang_cc1 -fmodules-cache-path=%t -fmodules-strict-decluse -fmodules -fimplicit-module-maps -I %S/Inputs %s 2>&1 | FileCheck %s -check-prefix=DECLUSE
// DECLUSE: error: module incomplete_mod does not depend on a module exporting {{'.*incomplete_mod_missing.h'}}
