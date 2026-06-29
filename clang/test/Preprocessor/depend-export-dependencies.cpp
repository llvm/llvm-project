// RUN: %clang_cc1 %s -fsyntax-only -verify -std=c++2d -MMD -MT %s.o --embed-dir=%S/Inputs -dependency-file - | FileCheck %s

// "export" has no meaing until p1130 makes headway, but this is parsed
// and should not affect anything nonetheless
#depend export <media/*>
#depend export "single_byte.txt"
// expected-no-diagnostics

// CHECK: depend.cpp.o
// CHECK-NEXT: depend.cpp
// CHECK-NEXT: Inputs{{[/\\]}}media{{[/\\]}}art.txt
// CHECK-NEXT: Inputs{{[/\\]}}media{{[/\\]}}empty
// CHECK-NEXT: single_byte.txt
