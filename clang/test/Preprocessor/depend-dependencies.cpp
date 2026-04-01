// RUN: %clang_cc1 %s -fsyntax-only -verify -std=c++2d -MMD -MT %s.o --embed-dir=%S/Inputs -dependency-file - | FileCheck %s

#depend <media/*>
#depend "single_byte.txt"
// expected-no-diagnostics

// CHECK: depend.cpp.o
// CHECK-NEXT: depend.cpp
// CHECK-NEXT: Inputs{{[/\\]}}media{{[/\\]}}art.txt
// CHECK-NEXT: Inputs{{[/\\]}}media{{[/\\]}}empty
// CHECK-NEXT: single_byte.txt
