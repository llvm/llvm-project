// RUN: %clang_cc1 -fsyntax-only %s 2>&1 | FileCheck %s

#line 10 "macro-directive-dump.c"
#define MACRO 1
#undef MACRO
#pragma clang __debug macro MACRO
// CHECK: UndefMacroDirective {{.*}} <macro-directive-dump.c:11:8>
// CHECK: DefMacroDirective {{.*}} <macro-directive-dump.c:10:9>

#line 20 "macro-directive-dump.c"
#define MACRO 1
#pragma push_macro("MACRO")
#define MACRO 2
#pragma pop_macro("MACRO")
#pragma clang __debug macro MACRO
// CHECK: DefMacroDirective {{.*}} <macro-directive-dump.c:23:9>
// CHECK: #define <macro> 1
// CHECK: UndefMacroDirective {{.*}} <macro-directive-dump.c:23:9>
// CHECK: DefMacroDirective {{.*}} <macro-directive-dump.c:20:9>
