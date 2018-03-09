// RUN: %clang_analyze_cc1 -analyzer-checker=debug.AnalysisOrder -analyzer-config c++-allocator-inlining=true,debug.AnalysisOrder:PreStmtCXXNewExpr=true,debug.AnalysisOrder:PostStmtCXXNewExpr=true,debug.AnalysisOrder:PreCall=true,debug.AnalysisOrder:PostCall=true,debug.AnalysisOrder:NewAllocator=true %s 2>&1 | FileCheck %s

#include "Inputs/system-header-simulator-cxx.h"

namespace std {
  void *malloc(size_t);
}

void *operator new(size_t size) { return std::malloc(size); }

struct S {
  S() {}
};

void foo();

void test() {
  S *s = new S();
  foo();
}

// CHECK:      PreCall (operator new)
// CHECK-NEXT: PreCall (std::malloc)
// CHECK-NEXT: PostCall (std::malloc)
// CHECK-NEXT: PostCall (operator new)
// CHECK-NEXT: NewAllocator
// CHECK-NEXT: PreCall (S::S)
// CHECK-NEXT: PostCall (S::S)
// CHECK-NEXT: PreStmt<CXXNewExpr>
// CHECK-NEXT: PostStmt<CXXNewExpr>
// CHECK-NEXT: PreCall (foo)
// CHECK-NEXT: PostCall (foo)
