// RUN: %clang_cc1 %s -fsyntax-only -verify -std=c++2d -MMD -MT %s.o --embed-dir=%S/Inputs -dependency-file - | FileCheck %s

#depend "media/**"
#depend <jk.txt>
// expected-no-diagnostics

// CHECK: depend-recursive.cpp.o
// CHECK-NEXT: depend-recursive.cpp
// CHECK-NEXT: Inputs{{[/\\]}}media{{[/\\]}}nested{{[/\\]}}inside.txt
// CHECK-NEXT: Inputs{{[/\\]}}media{{[/\\]}}art.txt
// CHECK-NEXT: Inputs{{[/\\]}}media{{[/\\]}}empty
// CHECK-NEXT: Inputs{{[/\\]}}jk.txt
