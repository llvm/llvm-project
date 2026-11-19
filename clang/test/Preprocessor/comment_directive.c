// RUN: %clang_cc1 -E -C %s | FileCheck %s

/* comment */ #define A 1
int a = A;

/*
 * multiline comment
 */ #define B 2
int b = B;

int c; /* comment */ #define C 3
int d = C;

int e; /*
 * multiline comment
 */ #define D 4
int f = D;

// CHECK: /* comment */
// CHECK-NEXT: int a = 1;

// CHECK: /*
// CHECK-NEXT:  * multiline comment
// CHECK-NEXT:  */
// CHECK-NEXT: int b = 2;

// CHECK: int c; /* comment */ #define C 3
// CHECK-NEXT: int d = C;

// CHECK: int e; /*
// CHECK-NEXT:  * multiline comment
// CHECK-NEXT:  */ #define D 4
// CHECK-NEXT: int f = D;
