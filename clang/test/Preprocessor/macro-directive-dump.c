// RUN: %clang_cc1 -fsyntax-only %s 2>&1 | FileCheck %s

#line 10 "macro-directive-dump.c"
#define MACRO 1
#undef MACRO
#pragma clang __debug macro MACRO

// CHECK: UndefMacroDirective {{.*}} <macro-directive-dump.c:11:8>
// CHECK: DefMacroDirective {{.*}} <macro-directive-dump.c:10:9>
