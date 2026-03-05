// RUN: cat %s | sed 's|@TESTDIR@|%/S|g' | clang-repl -Xcc -I%S/Inputs | FileCheck %s
extern "C" int printf(const char *, ...);
int x1 = 0;
int x2 = 42;
%undo
int x2 = 24;
auto r1 = printf("x1 = %d\n", x1); 
// CHECK: x1 = 0
auto r2 = printf("x2 = %d\n", x2);
// CHECK-NEXT: x2 = 24

int foo() { return 1; }
%undo
int foo() { return 2; }
auto r3 = printf("foo() = %d\n", foo());
// CHECK-NEXT: foo() = 2

inline int bar() { return 42;}
auto r4 = bar();
%undo
auto r5 = bar();

#include <cstdio>

#include "dynamic-header.h"
auto r6 = printf("getDynamicValue() = %d\n", getDynamicValue());
%undo
%undo
// CHECK-NEXT: getDynamicValue() = 100

FILE *f = fopen("@TESTDIR@/Inputs/dynamic-header.h", "w");

fprintf(f, "#ifndef DYNAMIC_HEADER_H\n");
fprintf(f, "#define DYNAMIC_HEADER_H\n");
fprintf(f, "inline int getDynamicValue() { return 200; }\n");
fprintf(f, "#endif\n");
fclose(f); 

#include "dynamic-header.h"
auto r8 = printf("getDynamicValue() = %d\n", getDynamicValue());
// CHECK-NEXT: getDynamicValue() = 200

%quit
