// RUN: %clang_cc1 -E -frewrite-includes -DFIRST -I %S/Inputs -I %S/Inputs/NextIncludes %s -o - | FileCheck -strict-whitespace %s
// RUN: %clang_cc1 -E -frewrite-includes -P -DFIRST -I %S/Inputs -I %S/Inputs/NextIncludes %s -o - | FileCheck -check-prefix=CHECKNL -strict-whitespace %s
// RUN: %clang_cc1 -E -frewrite-includes -DFIRST -I %S/Inputs -I %S/Inputs/NextIncludes %s -o - | %clang_cc1 -Wall -Wextra -Wconversion -DFIRST -x c -fsyntax-only 2>&1 | FileCheck -check-prefix=COMPILE --implicit-check-not warning: %s
// STARTCOMPARE
#define A(a,b) a ## b
A(in,t) a;
#include "rewrite-includes1.h"
#ifdef FIRST
#define HEADER "rewrite-includes3.h"
#include HEADER
#else
#include "rewrite-includes4.h"
#endif
  // indented
#/**/include /**/ "rewrite-includes5.h" /**/ \
 
#include "rewrite-includes6.h" // comment
 
#include "rewrite-includes6.h" /* comment
                                  continues */
#include "rewrite-includes7.h"
#include "rewrite-includes7.h"
#include "rewrite-includes8.h"
#include "rewrite-includes9.h"
static int unused;
// ENDCOMPARE
// CHECK: {{^}}# 1 "{{.*}}rewrite-includes.c"{{$}}
// CHECK: {{^}}// STARTCOMPARE{{$}}
// CHECK-NEXT: {{^}}#define A(a,b) a ## b{{$}}
// CHECK-NEXT: {{^}}A(in,t) a;{{$}}
// CHECK-NEXT: {{^}}#if defined(__CLANG_REWRITTEN_INCLUDES) /* rewrite-includes1.h expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#include "rewrite-includes1.h"{{$}}
// CHECK-NEXT: {{^}}#else /* rewrite-includes1.h expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 7 "{{.*}}rewrite-includes.c"{{$}}
// CHECK-NEXT: {{^}}# 1 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes1.h" 1{{$}}
// CHECK-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#pragma clang system_header{{$}}
// CHECK-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 2 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes1.h" 3{{$}}
// CHECK-NEXT: {{^}}int included_line1;{{$}}
// CHECK-NEXT: {{^}}#if defined(__CLANG_REWRITTEN_INCLUDES) || defined(__CLANG_REWRITTEN_SYSTEM_INCLUDES) /* rewrite-includes2.h expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#include "rewrite-includes2.h"{{$}}
// CHECK-NEXT: {{^}}#else /* rewrite-includes2.h expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 3 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes1.h" 3{{$}}
// CHECK-NEXT: {{^}}# 1 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes2.h" 1 3{{$}}
// CHECK-NEXT: {{^}}int included_line2;{{$}}
// CHECK-NEXT: {{^}}#endif /* rewrite-includes2.h expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 4 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes1.h" 2 3{{$}}
// CHECK-NEXT: {{^}}#endif /* rewrite-includes1.h expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 8 "{{.*}}rewrite-includes.c" 2{{$}}
// CHECK-NEXT: {{^}}#ifdef FIRST{{$}}
// CHECK-NEXT: {{^}}#define HEADER "rewrite-includes3.h"{{$}}
// CHECK-NEXT: {{^}}#if defined(__CLANG_REWRITTEN_INCLUDES) /* rewrite-includes3.h expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#include HEADER{{$}}
// CHECK-NEXT: {{^}}#else /* rewrite-includes3.h expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 10 "{{.*}}rewrite-includes.c"{{$}}
// CHECK-NEXT: {{^}}# 1 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes3.h" 1{{$}}
// CHECK-NEXT: {{^}}unsigned int included_line3 = -10;{{$}}
// CHECK-NEXT: {{^}}#endif /* rewrite-includes3.h expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 11 "{{.*}}rewrite-includes.c" 2{{$}}
// CHECK-NEXT: {{^}}#else{{$}}
// CHECK-NEXT: {{^}}# 12 "{{.*}}rewrite-includes.c"{{$}}
// CHECK-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#include "rewrite-includes4.h"{{$}}
// CHECK-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 12 "{{.*}}rewrite-includes.c"{{$}}
// CHECK-NEXT: {{^}}# 13 "{{.*}}rewrite-includes.c"{{$}}
// CHECK-NEXT: {{^}}#endif{{$}}
// CHECK-NEXT: {{^}}# 14 "{{.*}}rewrite-includes.c"{{$}}
// CHECK-NEXT: {{^}}  // indented{{$}}
// CHECK-NEXT: {{^}}#if defined(__CLANG_REWRITTEN_INCLUDES) /* rewrite-includes5.h expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#/**/include /**/ "rewrite-includes5.h" /**/ {{\\}}{{$}}
// CHECK-NEXT: {{^}} {{$}}
// CHECK-NEXT: {{^}}#else /* rewrite-includes5.h expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 16 "{{.*}}rewrite-includes.c"{{$}}
// CHECK-NEXT: {{^}}# 1 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes5.h" 1{{$}}
// CHECK-NEXT: {{^}}int included_line5;{{$}}
// CHECK-NEXT: {{^}}#endif /* rewrite-includes5.h expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 17 "{{.*}}rewrite-includes.c" 2{{$}}
// CHECK-NEXT: {{^}}#if defined(__CLANG_REWRITTEN_INCLUDES) /* rewrite-includes6.h expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#include "rewrite-includes6.h" // comment{{$}}
// CHECK-NEXT: {{^}}#else /* rewrite-includes6.h expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 17 "{{.*}}rewrite-includes.c"{{$}}
// CHECK-NEXT: {{^}}# 1 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes6.h" 1{{$}}
// CHECK-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#pragma once{{$}}
// CHECK-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 2 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes6.h"{{$}}
// CHECK-NEXT: {{^}}int included_line6;{{$}}
// CHECK-NEXT: {{^}}#endif /* rewrite-includes6.h expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 18 "{{.*}}rewrite-includes.c" 2{{$}}
// CHECK-NEXT: {{^}} {{$}}
// CHECK-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#include "rewrite-includes6.h" /* comment{{$}}
// CHECK-NEXT: {{^}}                                  continues */{{$}}
// CHECK-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 20 "{{.*}}rewrite-includes.c"{{$}}
// CHECK-NEXT: {{^}}# 21 "{{.*}}rewrite-includes.c"{{$}}
// CHECK-NEXT: {{^}}#if defined(__CLANG_REWRITTEN_INCLUDES) /* rewrite-includes7.h expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#include "rewrite-includes7.h"{{$}}
// CHECK-NEXT: {{^}}#else /* rewrite-includes7.h expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 21 "{{.*}}rewrite-includes.c"{{$}}
// CHECK-NEXT: {{^}}# 1 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes7.h" 1{{$}}
// CHECK-NEXT: {{^}}#ifndef REWRITE_INCLUDES_7{{$}}
// CHECK-NEXT: {{^}}#define REWRITE_INCLUDES_7{{$}}
// CHECK-NEXT: {{^}}int included_line7;{{$}}
// CHECK-NEXT: {{^}}#endif{{$}}
// CHECK-NEXT: {{^}}# 5 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes7.h"{{$}}
// CHECK-NEXT: {{^}}#endif /* rewrite-includes7.h expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 22 "{{.*}}rewrite-includes.c" 2{{$}}
// CHECK-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#include "rewrite-includes7.h"{{$}}
// CHECK-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 22 "{{.*}}rewrite-includes.c"{{$}}
// CHECK-NEXT: {{^}}# 23 "{{.*}}rewrite-includes.c"{{$}}
// CHECK-NEXT: {{^}}#if defined(__CLANG_REWRITTEN_INCLUDES) /* rewrite-includes8.h expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#include "rewrite-includes8.h"{{$}}
// CHECK-NEXT: {{^}}#else /* rewrite-includes8.h expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 23 "{{.*}}rewrite-includes.c"{{$}}
// CHECK-NEXT: {{^}}# 1 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes8.h" 1{{$}}
// CHECK-NEXT: {{^}}#if 0 /* disabled by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#if __has_include_next(<rewrite-includes8.h>){{$}}
// CHECK-NEXT: {{^}}#endif{{$}}
// CHECK-NEXT: {{^}}#endif /* disabled by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#if 0 /* evaluated by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 2 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes8.h"{{$}}
// CHECK-NEXT: {{^}}#if 0 /* disabled by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#if 0{{$}}
// CHECK-NEXT: {{^}}#elif __has_include(<rewrite-includes8.hfail>){{$}}
// CHECK-NEXT: {{^}}#endif{{$}}
// CHECK-NEXT: {{^}}#endif /* disabled by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#elif 0 /* evaluated by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 3 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes8.h"{{$}}
// CHECK-NEXT: {{^}}#endif{{$}}
// CHECK-NEXT: {{^}}# 4 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes8.h"{{$}}
// CHECK-NEXT: {{^}}#if 0 /* disabled by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#if !__has_include("rewrite-includes8.h"){{$}}
// CHECK-NEXT: {{^}}#endif{{$}}
// CHECK-NEXT: {{^}}#endif /* disabled by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#if 0 /* evaluated by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 5 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes8.h"{{$}}
// CHECK-NEXT: {{^}}#endif{{$}}
// CHECK-NEXT: {{^}}# 6 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes8.h"{{$}}
// CHECK-NEXT: {{^}}#endif /* rewrite-includes8.h expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 24 "{{.*}}rewrite-includes.c" 2{{$}}
// CHECK-NEXT: {{^}}#if defined(__CLANG_REWRITTEN_INCLUDES) /* rewrite-includes9.h expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#include "rewrite-includes9.h"{{$}}
// CHECK-NEXT: {{^}}#else /* rewrite-includes9.h expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 24 "{{.*}}rewrite-includes.c"{{$}}
// CHECK-NEXT: {{^}}# 1 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes9.h" 1{{$}}
// CHECK-NEXT: {{^}}#if 0 /* disabled by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#if __has_include_next(<rewrite-includes9.h>){{$}}
// CHECK-NEXT: {{^}}#endif{{$}}
// CHECK-NEXT: {{^}}#endif /* disabled by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#if 1 /* evaluated by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 2 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes9.h"{{$}}
// CHECK-NEXT: {{^}}#if defined(__CLANG_REWRITTEN_INCLUDES) /* rewrite-includes9.h expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#include_next <rewrite-includes9.h>{{$}}
// CHECK-NEXT: {{^}}#else /* rewrite-includes9.h expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 2 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes9.h"{{$}}
// CHECK-NEXT: {{^}}# 1 "{{.*[/\\]Inputs(/|\\\\)NextIncludes(/|\\\\)}}rewrite-includes9.h" 1{{$}}
// CHECK-NEXT: {{^}}int included_line9;{{$}}
// CHECK-NEXT: {{^}}#endif /* rewrite-includes9.h expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 3 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes9.h" 2{{$}}
// CHECK-NEXT: {{^}}#endif{{$}}
// CHECK-NEXT: {{^}}# 4 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes9.h"{{$}}
// CHECK-NEXT: {{^}}#endif /* rewrite-includes9.h expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 25 "{{.*}}rewrite-includes.c" 2{{$}}
// CHECK-NEXT: {{^}}static int unused;{{$}}
// CHECK-NEXT: {{^}}// ENDCOMPARE{{$}}

// CHECKNL: {{^}}// STARTCOMPARE{{$}}
// CHECKNL-NEXT: {{^}}#define A(a,b) a ## b{{$}}
// CHECKNL-NEXT: {{^}}A(in,t) a;{{$}}
// CHECKNL-NEXT: {{^}}#if defined(__CLANG_REWRITTEN_INCLUDES) /* rewrite-includes1.h expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#include "rewrite-includes1.h"{{$}}
// CHECKNL-NEXT: {{^}}#else /* rewrite-includes1.h expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#pragma clang system_header{{$}}
// CHECKNL-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}int included_line1;{{$}}
// CHECKNL-NEXT: {{^}}#if defined(__CLANG_REWRITTEN_INCLUDES) || defined(__CLANG_REWRITTEN_SYSTEM_INCLUDES) /* rewrite-includes2.h expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#include "rewrite-includes2.h"{{$}}
// CHECKNL-NEXT: {{^}}#else /* rewrite-includes2.h expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}int included_line2;{{$}}
// CHECKNL-NEXT: {{^}}#endif /* rewrite-includes2.h expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#endif /* rewrite-includes1.h expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#ifdef FIRST{{$}}
// CHECKNL-NEXT: {{^}}#define HEADER "rewrite-includes3.h"{{$}}
// CHECKNL-NEXT: {{^}}#if defined(__CLANG_REWRITTEN_INCLUDES) /* rewrite-includes3.h expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#include HEADER{{$}}
// CHECKNL-NEXT: {{^}}#else /* rewrite-includes3.h expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}unsigned int included_line3 = -10;{{$}}
// CHECKNL-NEXT: {{^}}#endif /* rewrite-includes3.h expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#else{{$}}
// CHECKNL-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#include "rewrite-includes4.h"{{$}}
// CHECKNL-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#endif{{$}}
// CHECKNL-NEXT: {{^}}  // indented{{$}}
// CHECKNL-NEXT: {{^}}#if defined(__CLANG_REWRITTEN_INCLUDES) /* rewrite-includes5.h expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#/**/include /**/ "rewrite-includes5.h" /**/ {{\\}}{{$}}
// CHECKNL-NEXT: {{^}} {{$}}
// CHECKNL-NEXT: {{^}}#else /* rewrite-includes5.h expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}int included_line5;{{$}}
// CHECKNL-NEXT: {{^}}#endif /* rewrite-includes5.h expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#if defined(__CLANG_REWRITTEN_INCLUDES) /* rewrite-includes6.h expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#include "rewrite-includes6.h" // comment{{$}}
// CHECKNL-NEXT: {{^}}#else /* rewrite-includes6.h expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#pragma once{{$}}
// CHECKNL-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}int included_line6;{{$}}
// CHECKNL-NEXT: {{^}}#endif /* rewrite-includes6.h expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}} {{$}}
// CHECKNL-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#include "rewrite-includes6.h" /* comment{{$}}
// CHECKNL-NEXT: {{^}}                                  continues */{{$}}
// CHECKNL-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#if defined(__CLANG_REWRITTEN_INCLUDES) /* rewrite-includes7.h expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#include "rewrite-includes7.h"{{$}}
// CHECKNL-NEXT: {{^}}#else /* rewrite-includes7.h expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#ifndef REWRITE_INCLUDES_7{{$}}
// CHECKNL-NEXT: {{^}}#define REWRITE_INCLUDES_7{{$}}
// CHECKNL-NEXT: {{^}}int included_line7;{{$}}
// CHECKNL-NEXT: {{^}}#endif{{$}}
// CHECKNL-NEXT: {{^}}#endif /* rewrite-includes7.h expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#include "rewrite-includes7.h"{{$}}
// CHECKNL-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#if defined(__CLANG_REWRITTEN_INCLUDES) /* rewrite-includes8.h expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#include "rewrite-includes8.h"{{$}}
// CHECKNL-NEXT: {{^}}#else /* rewrite-includes8.h expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#if 0 /* disabled by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#if __has_include_next(<rewrite-includes8.h>){{$}}
// CHECKNL-NEXT: {{^}}#endif{{$}}
// CHECKNL-NEXT: {{^}}#endif /* disabled by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#if 0 /* evaluated by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#if 0 /* disabled by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#if 0{{$}}
// CHECKNL-NEXT: {{^}}#elif __has_include(<rewrite-includes8.hfail>){{$}}
// CHECKNL-NEXT: {{^}}#endif{{$}}
// CHECKNL-NEXT: {{^}}#endif /* disabled by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#elif 0 /* evaluated by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#endif{{$}}
// CHECKNL-NEXT: {{^}}#if 0 /* disabled by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#if !__has_include("rewrite-includes8.h"){{$}}
// CHECKNL-NEXT: {{^}}#endif{{$}}
// CHECKNL-NEXT: {{^}}#endif /* disabled by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#if 0 /* evaluated by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#endif{{$}}
// CHECKNL-NEXT: {{^}}#endif /* rewrite-includes8.h expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#if defined(__CLANG_REWRITTEN_INCLUDES) /* rewrite-includes9.h expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#include "rewrite-includes9.h"{{$}}
// CHECKNL-NEXT: {{^}}#else /* rewrite-includes9.h expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#if 0 /* disabled by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#if __has_include_next(<rewrite-includes9.h>){{$}}
// CHECKNL-NEXT: {{^}}#endif{{$}}
// CHECKNL-NEXT: {{^}}#endif /* disabled by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#if 1 /* evaluated by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#if defined(__CLANG_REWRITTEN_INCLUDES) /* rewrite-includes9.h expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#include_next <rewrite-includes9.h>{{$}}
// CHECKNL-NEXT: {{^}}#else /* rewrite-includes9.h expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}int included_line9;{{$}}
// CHECKNL-NEXT: {{^}}#endif /* rewrite-includes9.h expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#endif{{$}}
// CHECKNL-NEXT: {{^}}#endif /* rewrite-includes9.h expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}static int unused;{{$}}
// CHECKNL-NEXT: {{^}}// ENDCOMPARE{{$}}

// COMPILE: Inputs{{[/\\]}}rewrite-includes3.h:1:31: warning: implicit conversion changes signedness:
// COMPILE: rewrite-includes.c:25:12: warning: unused variable 'unused'
