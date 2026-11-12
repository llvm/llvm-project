// RUN: %clang_cc1 -E -frewrite-includes -I %S/Inputs %s -o - | FileCheck -strict-whitespace %s
// RUN: %clang_cc1 -E -frewrite-includes -I %S/Inputs %s -o - | %clang_cc1 -Wall -Wextra -Wconversion -x c -fsyntax-only 2>&1 | FileCheck -check-prefix=COMPILE --implicit-check-not warning: %s

#define value1 1
#if value1
int line1;
#else
int line2;
#endif

#define value2 2

#if value1 == value2
int line3;
#elif value1 > value2
int line4;
#elif value1 < value2
int line5;
#else
int line6;
#endif

#if __has_include(<rewrite-includes3.h>)
#include <rewrite-includes3.h>
#endif

#define HAS_INCLUDE(x) __has_include(x)

#if HAS_INCLUDE(<rewrite-includes1.h>)
#endif

/*
#if value1
commented out
*/

#if value1 < value2 \
|| value1 != value2
int line7;
#endif

#if value1 /*
*/
#endif

static int unused;

// ENDCOMPARE

// CHECK: #if 0 /* disabled by -frewrite-includes */
// CHECK-NEXT: #if value1
// CHECK-NEXT: #endif
// CHECK-NEXT: #endif /* disabled by -frewrite-includes */
// CHECK-NEXT: #if 1 /* evaluated by -frewrite-includes */
// CHECK-NEXT: # 6 "{{.*}}rewrite-includes-conditions.c"

// CHECK: #if 0 /* disabled by -frewrite-includes */
// CHECK-NEXT: #if value1 == value2
// CHECK-NEXT: #endif
// CHECK-NEXT: #endif /* disabled by -frewrite-includes */
// CHECK-NEXT: #if 0 /* evaluated by -frewrite-includes */
// CHECK-NEXT: # 14 "{{.*}}rewrite-includes-conditions.c"

// CHECK: #if 0 /* disabled by -frewrite-includes */
// CHECK-NEXT: #if 0
// CHECK-NEXT: #elif value1 > value2
// CHECK-NEXT: #endif
// CHECK-NEXT: #endif /* disabled by -frewrite-includes */
// CHECK-NEXT: #elif 0 /* evaluated by -frewrite-includes */
// CHECK-NEXT: # 16 "{{.*}}rewrite-includes-conditions.c"

// CHECK: #if 0 /* disabled by -frewrite-includes */
// CHECK-NEXT: #if 0
// CHECK-NEXT: #elif value1 < value2
// CHECK-NEXT: #endif
// CHECK-NEXT: #endif /* disabled by -frewrite-includes */
// CHECK-NEXT: #elif 1 /* evaluated by -frewrite-includes */
// CHECK-NEXT: # 18 "{{.*}}rewrite-includes-conditions.c"

// CHECK: #if 0 /* disabled by -frewrite-includes */
// CHECK-NEXT: #if __has_include(<rewrite-includes3.h>)
// CHECK-NEXT: #endif
// CHECK-NEXT: #endif /* disabled by -frewrite-includes */
// CHECK-NEXT: #if 1 /* evaluated by -frewrite-includes */
// CHECK-NEXT: # 24 "{{.*}}rewrite-includes-conditions.c"

// CHECK: #if 0 /* disabled by -frewrite-includes */
// CHECK-NEXT: #if HAS_INCLUDE(<rewrite-includes1.h>)
// CHECK-NEXT: #endif
// CHECK-NEXT: #endif /* disabled by -frewrite-includes */
// CHECK-NEXT: #if 1 /* evaluated by -frewrite-includes */
// CHECK-NEXT: # 30 "{{.*}}rewrite-includes-conditions.c"

// CHECK: #if 0 /* disabled by -frewrite-includes */
// CHECK-NEXT: #if value1 < value2 \
// CHECK-NEXT: || value1 != value2
// CHECK-NEXT: #endif
// CHECK-NEXT: #endif /* disabled by -frewrite-includes */
// CHECK-NEXT: #if 1 /* evaluated by -frewrite-includes */
// CHECK-NEXT: # 39 "{{.*}}rewrite-includes-conditions.c"

// CHECK: #if 0 /* disabled by -frewrite-includes */
// CHECK-NEXT: #if value1 /*
// CHECK-NEXT: */
// CHECK-NEXT: #endif
// CHECK-NEXT: #endif /* disabled by -frewrite-includes */
// CHECK-NEXT: #if 1 /* evaluated by -frewrite-includes */
// CHECK-NEXT: # 44 "{{.*}}rewrite-includes-conditions.c"

// CHECK: {{^}}// ENDCOMPARE{{$}}

// COMPILE: Inputs{{[/\\]}}rewrite-includes3.h:1:31: warning: implicit conversion changes signedness:
// COMPILE: rewrite-includes-conditions.c:46:12: warning: unused variable 'unused'
