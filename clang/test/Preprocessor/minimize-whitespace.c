// RUN: %clang_cc1 -fminimize-whitespace -E %s 2>&1 | FileCheck %s --strict-whitespace --check-prefix=MINCOL
// RUN: %clang_cc1 -fminimize-whitespace -E -C %s 2>&1 | FileCheck %s --strict-whitespace --check-prefix=MINCCOL
// RUN: %clang_cc1 -fminimize-whitespace -E -P %s 2>&1 | FileCheck %s --strict-whitespace --check-prefix=MINWS
// RUN: %clang_cc1 -fminimize-whitespace -E -C -P %s 2>&1 | FileCheck %s --strict-whitespace --check-prefix=MINCWS
// The follow empty lines ensure that a #line directive is emitted instead of newline padding after the RUN comments.






#define NOT_OMP  omp  something
#define HASH #

  int  a;     /*  span-comment  */
  int  b  ;   //  line-comment
  _Pragma  (  "omp  barrier"  ) x //  more line-comments
  #pragma  omp  nothing  //  another comment
HASH  pragma  NOT_OMP
  int  e;    // again a line
  int  \
f  ;


// MINCOL:      {{^}}# 15 "{{.*}}minimize-whitespace.c"{{$}}
// MINCOL:      {{^}}int a;{{$}}
// MINCOL-NEXT: {{^}}int b;{{$}}
// MINCOL-NEXT: {{^}}#pragma omp barrier{{$}}
// MINCOL-NEXT: # 17 "{{.*}}minimize-whitespace.c"
// MINCOL-NEXT: {{^}}x{{$}}
// MINCOL-NEXT: {{^}}#pragma omp nothing{{$}}
// MINCOL-NEXT: {{^ }}#pragma omp something{{$}}
// MINCOL-NEXT: {{^}}int e;{{$}}
// MINCOL-NEXT: {{^}}int f;{{$}}

// FIXME: Comments after pragmas disappear, even without -fminimize-whitespace
// MINCCOL:      {{^}}# 15 "{{.*}}minimize-whitespace.c"{{$}}
// MINCCOL:      {{^}}int a;/*  span-comment  */{{$}}
// MINCCOL-NEXT: {{^}}int b;//  line-comment{{$}}
// MINCCOL-NEXT: {{^}}#pragma omp barrier{{$}}
// MINCCOL-NEXT: # 17 "{{.*}}minimize-whitespace.c"
// MINCCOL-NEXT: {{^}}x//  more line-comments{{$}}
// MINCCOL-NEXT: {{^}}#pragma omp nothing{{$}}
// MINCCOL-NEXT: {{^ }}#pragma omp something{{$}}
// MINCCOL-NEXT: {{^}}int e;// again a line{{$}}
// MINCCOL-NEXT: {{^}}int f;{{$}}

// MINWS:      {{^}}int a;int b;{{$}}
// MINWS-NEXT: {{^}}#pragma omp barrier{{$}}
// MINWS-NEXT: {{^}}x{{$}}
// MINWS-NEXT: {{^}}#pragma omp nothing{{$}}
// MINWS-NEXT: {{^ }}#pragma omp something int e;int f;{{$}}

// FIXME: Comments after pragmas disappear, even without -fminimize-whitespace
// MINCWS:      {{^}}int a;/*  span-comment  */int b;//  line-comment{{$}}
// MINCWS-NEXT: {{^}}#pragma omp barrier{{$}}
// MINCWS-NEXT: {{^}}x//  more line-comments{{$}}
// MINCWS-NEXT: {{^}}#pragma omp nothing{{$}}
// MINCWS-NEXT: {{^ }}#pragma omp something int e;// again a line{{$}}
// MINCWS-NEXT: {{^}}int f;

