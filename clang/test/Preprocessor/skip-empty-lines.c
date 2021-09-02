  int  a ;
  int  b ;
// A single empty line
  int  c ;
/*

more than 8 empty lines
(forces a line marker instead of newline padding)




*/
  int  d ;

// RUN: %clang_cc1 -E %s 2>&1 | FileCheck %s --strict-whitespace --check-prefix=LINEMARKERS
// RUN: %clang_cc1 -E -P %s 2>&1 | FileCheck %s --strict-whitespace --check-prefix=COLSONLY
// RUN: %clang_cc1 -E -fminimize-whitespace %s 2>&1 | FileCheck %s --strict-whitespace --check-prefix=MINCOL
// RUN: %clang_cc1 -E -P -fminimize-whitespace %s 2>&1 | FileCheck %s --strict-whitespace --check-prefix=MINWS

// Check behavior after varying number of lines without emitted tokens.

// LINEMARKERS:       {{^}}# 1 "{{.*}}skip-empty-lines.c" 2
// LINEMARKERS-NEXT: {{^}}  int a ;
// LINEMARKERS-NEXT: {{^}}  int b ;
// LINEMARKERS-EMPTY:
// LINEMARKERS-NEXT: {{^}}  int c ;
// LINEMARKERS-NEXT: {{^}}# 14 "{{.*}}skip-empty-lines.c"
// LINEMARKERS-NEXT: {{^}}  int d ;

// COLSONLY:      {{^}}  int a ;
// COLSONLY-NEXT: {{^}}  int b ;
// COLSONLY-NEXT: {{^}}  int c ;
// COLSONLY-NEXT: {{^}}  int d ;

// MINCOL:      {{^}}# 1 "{{.*}}skip-empty-lines.c" 2
// MINCOL-NEXT: {{^}}int a;
// MINCOL-NEXT: {{^}}int b;
// MINCOL-EMPTY:
// MINCOL-NEXT: {{^}}int c;
// MINCOL-NEXT: {{^}}# 14 "{{.*}}skip-empty-lines.c"
// MINCOL-NEXT: {{^}}int d;

// MINWS: {{^}}int a;int b;int c;int d;

