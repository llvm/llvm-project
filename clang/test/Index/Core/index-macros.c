// RUN: c-index-test core -print-source-symbols -- %s | FileCheck %s

// CHECK: [[@LINE+1]]:9 | macro/C | X1 | c:index-macros.c@157@macro@X1 | Def |
#define X1 1
// CHECK: [[@LINE+1]]:9 | macro/C | DEF | c:index-macros.c@251@macro@DEF | Def |
#define DEF(x) int x
// CHECK: [[@LINE+1]]:8 | macro/C | X1 | c:index-macros.c@157@macro@X1 | Undef |
#undef X1

// CHECK: [[@LINE+2]]:1 | macro/C | DEF | c:index-macros.c@251@macro@DEF | Ref |
// CHECK: [[@LINE+1]]:5 | variable/C | i | c:@i | {{.*}} | Def | rel: 0
DEF(i);
