// RUN: not %clang_cc1 -verify -E -frewrite-includes -DFIRST -I %S/Inputs -I %S/Inputs/NextIncludes %s -o - | FileCheck -strict-whitespace %s
// RUN: not %clang_cc1 -verify -E -frewrite-includes -P -DFIRST -I %S/Inputs -I %S/Inputs/NextIncludes %s -o - | FileCheck -check-prefix=CHECKNL -strict-whitespace %s
// STARTCOMPARE
#define A(a,b) a ## b
A(1,2)
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
// ENDCOMPARE
// CHECK: {{^}}# 1 "{{.*}}rewrite-includes.c"{{$}}
// CHECK: {{^}}// STARTCOMPARE{{$}}
// CHECK-NEXT: {{^}}#define A(a,b) a ## b{{$}}
// CHECK-NEXT: {{^}}A(1,2){{$}}
// CHECK-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#include "rewrite-includes1.h"{{$}}
// CHECK-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 6 "{{.*}}rewrite-includes.c"{{$}}
// CHECK-NEXT: {{^}}# 1 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes1.h" 1{{$}}
// CHECK-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#pragma clang system_header{{$}}
// CHECK-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 2 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes1.h" 3{{$}}
// CHECK-NEXT: {{^}}included_line1{{$}}
// CHECK-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#include "rewrite-includes2.h"{{$}}
// CHECK-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 3 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes1.h" 3{{$}}
// CHECK-NEXT: {{^}}# 1 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes2.h" 1 3{{$}}
// CHECK-NEXT: {{^}}included_line2{{$}}
// CHECK-NEXT: {{^}}# 4 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes1.h" 2 3{{$}}
// CHECK-NEXT: {{^}}# 7 "{{.*}}rewrite-includes.c" 2{{$}}
// CHECK-NEXT: {{^}}#ifdef FIRST{{$}}
// CHECK-NEXT: {{^}}#define HEADER "rewrite-includes3.h"{{$}}
// CHECK-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#include HEADER{{$}}
// CHECK-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 9 "{{.*}}rewrite-includes.c"{{$}}
// CHECK-NEXT: {{^}}# 1 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes3.h" 1{{$}}
// CHECK-NEXT: {{^}}included_line3{{$}}
// CHECK-NEXT: {{^}}# 10 "{{.*}}rewrite-includes.c" 2{{$}}
// CHECK-NEXT: {{^}}#else{{$}}
// CHECK-NEXT: {{^}}# 11 "{{.*}}rewrite-includes.c"{{$}}
// CHECK-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#include "rewrite-includes4.h"{{$}}
// CHECK-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 11 "{{.*}}rewrite-includes.c"{{$}}
// CHECK-NEXT: {{^}}# 12 "{{.*}}rewrite-includes.c"{{$}}
// CHECK-NEXT: {{^}}#endif{{$}}
// CHECK-NEXT: {{^}}# 13 "{{.*}}rewrite-includes.c"{{$}}
// CHECK-NEXT: {{^}}  // indented{{$}}
// CHECK-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#/**/include /**/ "rewrite-includes5.h" /**/ {{\\}}{{$}}
// CHECK-NEXT: {{^}} {{$}}
// CHECK-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 15 "{{.*}}rewrite-includes.c"{{$}}
// CHECK-NEXT: {{^}}# 1 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes5.h" 1{{$}}
// CHECK-NEXT: {{^}}included_line5{{$}}
// CHECK-NEXT: {{^}}# 16 "{{.*}}rewrite-includes.c" 2{{$}}
// CHECK-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#include "rewrite-includes6.h" // comment{{$}}
// CHECK-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 16 "{{.*}}rewrite-includes.c"{{$}}
// CHECK-NEXT: {{^}}# 1 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes6.h" 1{{$}}
// CHECK-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#pragma once{{$}}
// CHECK-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 2 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes6.h"{{$}}
// CHECK-NEXT: {{^}}included_line6{{$}}
// CHECK-NEXT: {{^}}# 17 "{{.*}}rewrite-includes.c" 2{{$}}
// CHECK-NEXT: {{^}} {{$}}
// CHECK-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#include "rewrite-includes6.h" /* comment{{$}}
// CHECK-NEXT: {{^}}                                  continues */{{$}}
// CHECK-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 19 "{{.*}}rewrite-includes.c"{{$}}
// CHECK-NEXT: {{^}}# 20 "{{.*}}rewrite-includes.c"{{$}}
// CHECK-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#include "rewrite-includes7.h"{{$}}
// CHECK-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 20 "{{.*}}rewrite-includes.c"{{$}}
// CHECK-NEXT: {{^}}# 1 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes7.h" 1{{$}}
// CHECK-NEXT: {{^}}#ifndef REWRITE_INCLUDES_7{{$}}
// CHECK-NEXT: {{^}}#define REWRITE_INCLUDES_7{{$}}
// CHECK-NEXT: {{^}}included_line7{{$}}
// CHECK-NEXT: {{^}}#endif{{$}}
// CHECK-NEXT: {{^}}# 5 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes7.h"{{$}}
// CHECK-NEXT: {{^}}# 21 "{{.*}}rewrite-includes.c" 2{{$}}
// CHECK-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#include "rewrite-includes7.h"{{$}}
// CHECK-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 21 "{{.*}}rewrite-includes.c"{{$}}
// CHECK-NEXT: {{^}}# 22 "{{.*}}rewrite-includes.c"{{$}}
// CHECK-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#include "rewrite-includes8.h"{{$}}
// CHECK-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 22 "{{.*}}rewrite-includes.c"{{$}}
// CHECK-NEXT: {{^}}# 1 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes8.h" 1{{$}}
// CHECK-NEXT: {{^}}#if (0)/*__has_include_next(<rewrite-includes8.h>)*/{{$}}
// CHECK-NEXT: {{^}}#elif (0)/*__has_include(<rewrite-includes8.hfail>)*/{{$}}
// CHECK-NEXT: {{^}}# 3 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes8.h"{{$}}
// CHECK-NEXT: {{^}}#endif{{$}}
// CHECK-NEXT: {{^}}# 4 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes8.h"{{$}}
// CHECK-NEXT: {{^}}#if !(1)/*__has_include("rewrite-includes8.h")*/{{$}}
// CHECK-NEXT: {{^}}#endif{{$}}
// CHECK-NEXT: {{^}}# 6 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes8.h"{{$}}
// CHECK-NEXT: {{^}}# 23 "{{.*}}rewrite-includes.c" 2{{$}}
// CHECK-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#include "rewrite-includes9.h"{{$}}
// CHECK-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 23 "{{.*}}rewrite-includes.c"{{$}}
// CHECK-NEXT: {{^}}# 1 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes9.h" 1{{$}}
// CHECK-NEXT: {{^}}#if (1)/*__has_include_next(<rewrite-includes9.h>)*/{{$}}
// CHECK-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}#include_next <rewrite-includes9.h>{{$}}
// CHECK-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: {{^}}# 2 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes9.h"{{$}}
// CHECK-NEXT: {{^}}# 1 "{{.*[/\\]Inputs(/|\\\\)NextIncludes(/|\\\\)}}rewrite-includes9.h" 1{{$}}
// CHECK-NEXT: {{^}}included_line9{{$}}
// CHECK-NEXT: {{^}}# 3 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes9.h" 2{{$}}
// CHECK-NEXT: {{^}}#endif{{$}}
// CHECK-NEXT: {{^}}# 4 "{{.*[/\\]Inputs(/|\\\\)}}rewrite-includes9.h"{{$}}
// CHECK-NEXT: {{^}}# 24 "{{.*}}rewrite-includes.c" 2{{$}}
// CHECK-NEXT: {{^}}// ENDCOMPARE{{$}}

// CHECKNL: {{^}}// STARTCOMPARE{{$}}
// CHECKNL-NEXT: {{^}}#define A(a,b) a ## b{{$}}
// CHECKNL-NEXT: {{^}}A(1,2){{$}}
// CHECKNL-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#include "rewrite-includes1.h"{{$}}
// CHECKNL-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#pragma clang system_header{{$}}
// CHECKNL-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}included_line1{{$}}
// CHECKNL-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#include "rewrite-includes2.h"{{$}}
// CHECKNL-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}included_line2{{$}}
// CHECKNL-NEXT: {{^}}#ifdef FIRST{{$}}
// CHECKNL-NEXT: {{^}}#define HEADER "rewrite-includes3.h"{{$}}
// CHECKNL-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#include HEADER{{$}}
// CHECKNL-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}included_line3{{$}}
// CHECKNL-NEXT: {{^}}#else{{$}}
// CHECKNL-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#include "rewrite-includes4.h"{{$}}
// CHECKNL-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#endif{{$}}
// CHECKNL-NEXT: {{^}}  // indented{{$}}
// CHECKNL-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#/**/include /**/ "rewrite-includes5.h" /**/ {{\\}}{{$}}
// CHECKNL-NEXT: {{^}} {{$}}
// CHECKNL-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}included_line5{{$}}
// CHECKNL-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#include "rewrite-includes6.h" // comment{{$}}
// CHECKNL-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#pragma once{{$}}
// CHECKNL-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}included_line6{{$}}
// CHECKNL-NEXT: {{^}} {{$}}
// CHECKNL-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#include "rewrite-includes6.h" /* comment{{$}}
// CHECKNL-NEXT: {{^}}                                  continues */{{$}}
// CHECKNL-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#include "rewrite-includes7.h"{{$}}
// CHECKNL-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#ifndef REWRITE_INCLUDES_7{{$}}
// CHECKNL-NEXT: {{^}}#define REWRITE_INCLUDES_7{{$}}
// CHECKNL-NEXT: {{^}}included_line7{{$}}
// CHECKNL-NEXT: {{^}}#endif{{$}}
// CHECKNL-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#include "rewrite-includes7.h"{{$}}
// CHECKNL-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#include "rewrite-includes8.h"{{$}}
// CHECKNL-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#if (0)/*__has_include_next(<rewrite-includes8.h>)*/{{$}}
// CHECKNL-NEXT: {{^}}#elif (0)/*__has_include(<rewrite-includes8.hfail>)*/{{$}}
// CHECKNL-NEXT: {{^}}#endif{{$}}
// CHECKNL-NEXT: {{^}}#if !(1)/*__has_include("rewrite-includes8.h")*/{{$}}
// CHECKNL-NEXT: {{^}}#endif{{$}}
// CHECKNL-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#include "rewrite-includes9.h"{{$}}
// CHECKNL-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#if (1)/*__has_include_next(<rewrite-includes9.h>)*/{{$}}
// CHECKNL-NEXT: {{^}}#if 0 /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}#include_next <rewrite-includes9.h>{{$}}
// CHECKNL-NEXT: {{^}}#endif /* expanded by -frewrite-includes */{{$}}
// CHECKNL-NEXT: {{^}}included_line9{{$}}
// CHECKNL-NEXT: {{^}}#endif{{$}}
// CHECKNL-NEXT: {{^}}// ENDCOMPARE{{$}}
