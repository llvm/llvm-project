// RUN: %clang_analyze_cc1 -analyzer-checker=debug.AnalysisOrder -analyzer-config c++-allocator-inlining=true,debug.AnalysisOrder:PreStmtCXXNewExpr=true,debug.AnalysisOrder:PostStmtCXXNewExpr=true,debug.AnalysisOrder:PreCall=true,debug.AnalysisOrder:PostCall=true,debug.AnalysisOrder:NewAllocator=true %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-INLINE
// RUN: %clang_analyze_cc1 -analyzer-checker=debug.AnalysisOrder -analyzer-config c++-allocator-inlining=false,debug.AnalysisOrder:PreStmtCXXNewExpr=true,debug.AnalysisOrder:PostStmtCXXNewExpr=true,debug.AnalysisOrder:PreCall=true,debug.AnalysisOrder:PostCall=true,debug.AnalysisOrder:NewAllocator=true %s 2>&1 | FileCheck %s  --check-prefixes=CHECK,CHECK-NO-INLINE

#include "Inputs/system-header-simulator-cxx.h"

namespace std {
void *malloc(size_t);
void free(void *);
} // namespace std

void *operator new(size_t size) { return std::malloc(size); }
void operator delete(void *ptr, size_t size) { std::free(ptr); }

struct S {
  S() {}
  ~S() {}
};

void foo();

void test() {
  S *s = new S();
  foo();
  delete s;
}

/*
void test() {
  S *s = new S();
// CHECK-INLINE:      PreCall (operator new)
// CHECK-INLINE-NEXT: PreCall (std::malloc)
// CHECK-INLINE-NEXT: PostCall (std::malloc)
// CHECK-INLINE-NEXT: PostCall (operator new)
// CHECK-INLINE-NEXT: NewAllocator
// CHECK-NO-INLINE: PreCall (S::S)
// CHECK-INLINE-NEXT: PreCall (S::S)
// CHECK-NEXT: PostCall (S::S)
// CHECK-NEXT: PreStmt<CXXNewExpr>
// CHECK-NEXT: PostStmt<CXXNewExpr>
  foo();
// CHECK-NEXT: PreCall (foo)
// CHECK-NEXT: PostCall (foo)
  delete s;
// CHECK-NEXT: PreCall (S::~S)
// CHECK-NEXT: PostCall (S::~S)
// CHECK-NEXT: PreCall (operator delete)
// CHECK-INLINE-NEXT: PreCall (std::free)
// CHECK-INLINE-NEXT: PostCall (std::free)
// CHECK-NEXT: PostCall (operator delete)
}

void operator delete(void *ptr, size_t size) {
  std::free(ptr);
// CHECK-NO-INLINE-NEXT: PreCall (std::free)
// CHECK-NO-INLINE-NEXT: PostCall (std::free)
}

void *operator new(size_t size) {
  return std::malloc(size);
// CHECK-NO-INLINE-NEXT: PreCall (std::malloc)
// CHECK-NO-INLINE-NEXT: PostCall (std::malloc)
}
*/
