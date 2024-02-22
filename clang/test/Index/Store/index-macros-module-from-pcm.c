// Compile without indexing first, then again with indexing which will index
// the .pcm itself.
// RUN: rm -rf %t-from-module
// RUN: %clang_cc1 %s -I %S/Inputs -fmodules -fimplicit-module-maps -fmodules-cache-path=%t-from-module/mcp
// RUN: %clang_cc1 %s -I %S/Inputs -fmodules -fimplicit-module-maps -fmodules-cache-path=%t-from-module/mcp -index-store-path %t-from-module/idx
// RUN: c-index-test core -print-record %t-from-module/idx | FileCheck %s

#pragma clang module import IndexMacrosMod
// Note: only the latest definition is found when indexing from the PCM file.

// CHECK-NOT: INDEXMACROSMODULE_H
// CHECK: macro/C | INDEX_MACROS_MODULE | [[USR2:.*@macro@INDEX_MACROS_MODULE]] | <no-cgname> | Def
// CHECK: 6:9 | macro/C | [[USR2]] | Def |
// CHECK-NOT: INDEXMACROSMODULE_H
