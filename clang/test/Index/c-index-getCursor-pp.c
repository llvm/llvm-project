#define OBSCURE(X) X
#define DECORATION
#define FNM(X) OBSCURE(X)
typedef int T;
void OBSCURE(func)(int x) {
  OBSCURE(T) DECORATION value;
}

#include "a.h"

#define A(X) X
#define B(X) A(X)

B(int x);

const char *fname = __FILE__;

#include <a.h>

#ifdef OBSCURE
#endif

#if defined(OBSCURE)
#endif

#define C(A) A

// RUN: c-index-test -cursor-at=%s:1:11 -I%S/Inputs %s | FileCheck -check-prefix=CHECK-1 %s
// CHECK-1: macro definition=OBSCURE
// RUN: c-index-test -cursor-at=%s:2:14 -I%S/Inputs %s | FileCheck -check-prefix=CHECK-2 %s
// CHECK-2: macro definition=DECORATION
// RUN: c-index-test -cursor-at=%s:5:7 -I%S/Inputs %s | FileCheck -check-prefix=CHECK-3 %s
// CHECK-3: macro expansion=OBSCURE:1:9
// RUN: c-index-test -cursor-at=%s:6:6 -I%S/Inputs %s | FileCheck -check-prefix=CHECK-4 %s
// CHECK-4: macro expansion=OBSCURE:1:9
// RUN: c-index-test -cursor-at=%s:6:19 -I%S/Inputs %s | FileCheck -check-prefix=CHECK-5 %s
// CHECK-5: macro expansion=DECORATION:2:9
// RUN: c-index-test -cursor-at=%s:9:10 -I%S/Inputs %s | FileCheck -check-prefix=CHECK-6 %s
// CHECK-6: inclusion directive=a.h
// RUN: c-index-test -cursor-at=%s:14:1 -I%S/Inputs %s | FileCheck -check-prefix=CHECK-7 %s
// CHECK-7: macro expansion=B:12:9
// RUN: c-index-test -cursor-at=%s:16:25 -I%S/Inputs %s | FileCheck -check-prefix=CHECK-8 %s
// CHECK-8: macro expansion=__FILE__
// RUN: c-index-test -cursor-at=%s:18:12 -I%S/Inputs %s | FileCheck -check-prefix=CHECK-9 %s
// CHECK-9: inclusion directive=a.h
// RUN: c-index-test -cursor-at=%s:20:10 -cursor-at=%s:23:15 -I%S/Inputs %s | FileCheck -check-prefix=CHECK-10 %s
// CHECK-10: 20:8 macro expansion=OBSCURE
// CHECK-10: 23:13 macro expansion=OBSCURE

// RUN: c-index-test -cursor-at=%s:3:20 -cursor-at=%s:12:14 \
// RUN:              -cursor-at=%s:26:11 -cursor-at=%s:26:14 -I%S/Inputs %s | FileCheck -check-prefix=CHECK-IN-MACRODEF %s
// CHECK-IN-MACRODEF: 3:16 macro expansion=OBSCURE
// CHECK-IN-MACRODEF: 12:14 macro expansion=A
// CHECK-IN-MACRODEF: 26:9 macro definition=C
// CHECK-IN-MACRODEF: 26:9 macro definition=C

// Same tests, but with "editing" optimizations
// RUN: env CINDEXTEST_EDITING=1 c-index-test -cursor-at=%s:1:11 -I%S/Inputs %s | FileCheck -check-prefix=CHECK-1 %s
// RUN: env CINDEXTEST_EDITING=1 c-index-test -cursor-at=%s:2:14 -I%S/Inputs %s | FileCheck -check-prefix=CHECK-2 %s
// RUN: env CINDEXTEST_EDITING=1 c-index-test -cursor-at=%s:5:7 -I%S/Inputs %s | FileCheck -check-prefix=CHECK-3 %s
// RUN: env CINDEXTEST_EDITING=1 c-index-test -cursor-at=%s:9:10 -I%S/Inputs %s | FileCheck -check-prefix=CHECK-6 %s
// RUN: env CINDEXTEST_EDITING=1 c-index-test -cursor-at=%s:3:20 -cursor-at=%s:12:14 \
// RUN:              -cursor-at=%s:26:11 -cursor-at=%s:26:14 -I%S/Inputs %s | FileCheck -check-prefix=CHECK-IN-MACRODEF %s
