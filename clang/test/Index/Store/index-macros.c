// RUN: rm -rf %t.idx
// RUN: %clang_cc1 %s -index-store-path %t.idx
// RUN: c-index-test core -print-record %t.idx | FileCheck %s

// RUN: rm -rf %t.idx.ignore
// RUN: %clang_cc1 %s -index-store-path %t.idx.ignore -index-ignore-macros
// RUN: c-index-test core -print-record %t.idx.ignore | FileCheck %s -check-prefix DISABLED
// DISABLED-NOT: macro/C
// DISABLED-NOT: X1

// CHECK: macro/C | X1 | [[X1_USR:.*@macro@X1]] | <no-cgname> | Def,Ref,Undef -
// CHECK: macro/C | DEF | [[DEF_USR:.*@macro@DEF]] | <no-cgname> | Def,Ref -
// CHECK: macro/C | REDEF | [[REDEF_USR1:.*@macro@REDEF]] | <no-cgname> | Def,Undef -
// CHECK: macro/C | REDEF | [[REDEF_USR2:.*@macro@REDEF]] | <no-cgname> | Def -

// CHECK: [[@LINE+1]]:9 | macro/C | [[X1_USR]] | Def |
#define X1 1
// CHECK: [[@LINE+1]]:9 | macro/C | [[DEF_USR]] | Def |
#define DEF(x) int x

// CHECK: [[@LINE+1]]:8 | macro/C | [[X1_USR]] | Ref
#ifdef X1
#endif

// CHECK: [[@LINE+1]]:8 | macro/C | [[X1_USR]] | Undef |
#undef X1

// CHECK: [[@LINE+1]]:9 | macro/C | [[REDEF_USR1]] | Def |
#define REDEF
// CHECK: [[@LINE+1]]:8 | macro/C | [[REDEF_USR1]] | Undef |
#undef REDEF
// CHECK: [[@LINE+1]]:9 | macro/C | [[REDEF_USR2]] | Def |
#define REDEF

// FIXME: index references to builtin macros. Adding a test since this was
// crashing at one point.
// CHECK-NOT: [[@LINE+1]]:5 | macro/C | __LINE__
#if __LINE__ == 41
#endif

// Macro references currently not supported.
// CHECK: [[@LINE+2]]:1 | macro/C | [[DEF_USR]] | Ref | rel: 0
// CHECK: [[@LINE+1]]:5 | variable/C | c:@i | Def |
DEF(i);
