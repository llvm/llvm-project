// RUN: c-index-test core -print-source-symbols -- %s | FileCheck %s
// RUN: c-index-test core -print-source-symbols -ignore-macros -- %s | FileCheck %s -check-prefix=DISABLED
// DISABLED-NOT: macro/C
// DISABLED-NOT: XI

// CHECK: [[@LINE+1]]:9 | macro/C | X1 | [[X1_USR:.*@macro@X1]] | Def |
#define X1 1
// CHECK: [[@LINE+1]]:9 | macro/C | DEF | [[DEF_USR:.*@macro@DEF]] | Def |
#define DEF(x) int x
// CHECK: [[@LINE+1]]:8 | macro/C | X1 | [[X1_USR]] | Undef |
#undef X1

// CHECK: [[@LINE+1]]:9 | macro/C | C | [[C_USR:.*@macro@C]] | Def |
#define C 1
// CHECK: [[@LINE+1]]:5 | macro/C | C | [[C_USR]] | Ref |
#if C
#endif
// CHECK: [[@LINE+1]]:8 | macro/C | C | [[C_USR]] | Ref |
#ifdef C
#endif
// CHECK: [[@LINE+1]]:9 | macro/C | C | [[C_USR]] | Ref |
#ifndef C
#endif
// CHECK: [[@LINE+1]]:13 | macro/C | C | [[C_USR]] | Ref |
#if defined(C)
#endif
// CHECK: [[@LINE+1]]:14 | macro/C | C | [[C_USR]] | Ref |
#if !defined(C)
#endif

// Nonexistent macros should not be included.
// CHECK-NOT: NOT_DEFINED
#ifdef NOT_DEFINED
#endif
#ifndef NOT_DEFINED
#endif
#if defined(NOT_DEFINED) && NOT_DEFINED
#elif !defined(NOT_DEFINED)
#endif

// CHECK: [[@LINE+1]]:5 | macro/C | __LINE__ | c:@macro@__LINE__ | Ref |
#if __LINE__ == 41
#endif

// CHECK: [[@LINE+2]]:1 | macro/C | DEF | [[DEF_USR]] | Ref |
// CHECK: [[@LINE+1]]:5 | variable/C | i | c:@i | {{.*}} | Def | rel: 0
DEF(i);
