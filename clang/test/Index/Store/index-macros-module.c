// RUN: rm -rf %t
// RUN: %clang_cc1 %s -I %S/Inputs -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/mcp -index-store-path %t/idx
// RUN: c-index-test core -print-record %t/idx | FileCheck %s

#pragma clang module import IndexMacrosMod
// CHECK: macro/C | INDEX_MACROS_MODULE | [[USR1:.*@macro@INDEX_MACROS_MODULE]] | <no-cgname> | Def,Undef -
// CHECK: macro/C | INDEX_MACROS_MODULE | [[USR2:.*@macro@INDEX_MACROS_MODULE]] | <no-cgname> | Def,Ref -

// CHECK-NOT: INDEXMACROSMODULE_H
// CHECK: 4:9 | macro/C | [[USR1]] | Def |
// CHECK: 5:8 | macro/C | [[USR1]] | Undef |
// CHECK: 6:9 | macro/C | [[USR2]] | Def |
// CHECK: 7:9 | macro/C | [[USR2]] | Ref |
// CHECK-NOT: INDEXMACROSMODULE_H
