// RUN: %clang_analyze_cc1 -std=c++20 -fblocks -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.AnalysisOrder \
// RUN:   -analyzer-config debug.AnalysisOrder:PreStmtCXXNewExpr=true \
// RUN:   -analyzer-config debug.AnalysisOrder:PostStmtCXXNewExpr=true \
// RUN:   -analyzer-config debug.AnalysisOrder:PreStmtCXXDeleteExpr=true \
// RUN:   -analyzer-config debug.AnalysisOrder:PostStmtCXXDeleteExpr=true \
// RUN:   -analyzer-config debug.AnalysisOrder:PreCall=true \
// RUN:   -analyzer-config debug.AnalysisOrder:PostCall=true \
// RUN:   2>&1 | FileCheck %s

// expected-no-diagnostics

#include "Inputs/system-header-simulator-cxx.h"

void f() {
  // C++20 standard draft 17.6.1.15:
  // Required behavior: A call to an operator delete with a size parameter may
  // be changed to a call to the corresponding operator delete without a size
  // parameter, without affecting memory allocation. [ Note: A conforming
  // implementation is for operator delete(void* ptr, size_t size) to simply
  // call operator delete(ptr). — end note ]
  //
  // C++20 standard draft 17.6.1.24, about nothrow operator delete:
  //   void operator delete(void* ptr, const std::nothrow_t&) noexcept;
  //   void operator delete(void* ptr, std::align_val_t alignment,
  //                        const std::nothrow_t&) noexcept;
  // Default behavior: Calls operator delete(ptr), or operator delete(ptr,
  // alignment), respectively.

  // FIXME: All calls to operator new should be CXXAllocatorCall, and calls to
  // operator delete should be CXXDeallocatorCall.
  {
    int *p = new int;
    delete p;
    // CHECK:      PreCall (operator new) [CXXAllocatorCall]
    // CHECK-NEXT: PostCall (operator new) [CXXAllocatorCall]
    // CHECK-NEXT: PreStmt<CXXNewExpr>
    // CHECK-NEXT: PostStmt<CXXNewExpr>
    // CHECK-NEXT: PreStmt<CXXDeleteExpr>
    // CHECK-NEXT: PostStmt<CXXDeleteExpr>
    // CHECK-NEXT: PreCall (operator delete) [CXXDeallocatorCall]
    // CHECK-NEXT: PostCall (operator delete) [CXXDeallocatorCall]

    p = new int;
    operator delete(p, 23542368);
    // CHECK-NEXT: PreCall (operator new) [CXXAllocatorCall]
    // CHECK-NEXT: PostCall (operator new) [CXXAllocatorCall]
    // CHECK-NEXT: PreStmt<CXXNewExpr>
    // CHECK-NEXT: PostStmt<CXXNewExpr>
    // CHECK-NEXT: PreCall (operator delete) [SimpleFunctionCall]
    // CHECK-NEXT: PostCall (operator delete) [SimpleFunctionCall]

    void *v = operator new(sizeof(int[2]), std::align_val_t(2));
    operator delete(v, std::align_val_t(2));
    // CHECK-NEXT: PreCall (operator new) [SimpleFunctionCall]
    // CHECK-NEXT: PostCall (operator new) [SimpleFunctionCall]
    // CHECK-NEXT: PreCall (operator delete) [SimpleFunctionCall]
    // CHECK-NEXT: PostCall (operator delete) [SimpleFunctionCall]

    v = operator new(sizeof(int[2]), std::align_val_t(2));
    operator delete(v, 345345, std::align_val_t(2));
    // CHECK-NEXT: PreCall (operator new) [SimpleFunctionCall]
    // CHECK-NEXT: PostCall (operator new) [SimpleFunctionCall]
    // CHECK-NEXT: PreCall (operator delete) [SimpleFunctionCall]
    // CHECK-NEXT: PostCall (operator delete) [SimpleFunctionCall]

    p = new (std::nothrow) int;
    operator delete(p, std::nothrow);
    // CHECK-NEXT: PreCall (operator new) [CXXAllocatorCall]
    // CHECK-NEXT: PostCall (operator new) [CXXAllocatorCall]
    // CHECK-NEXT: PreStmt<CXXNewExpr>
    // CHECK-NEXT: PostStmt<CXXNewExpr>
    // CHECK-NEXT: PreCall (operator delete) [SimpleFunctionCall]
    // CHECK-NEXT: PostCall (operator delete) [SimpleFunctionCall]

    v = operator new(sizeof(int[2]), std::align_val_t(2), std::nothrow);
    operator delete(v, std::align_val_t(2), std::nothrow);
    // CHECK-NEXT: PreCall (operator new) [SimpleFunctionCall]
    // CHECK-NEXT: PostCall (operator new) [SimpleFunctionCall]
    // CHECK-NEXT: PreCall (operator delete) [SimpleFunctionCall]
    // CHECK-NEXT: PostCall (operator delete) [SimpleFunctionCall]
  }

  {
    int *p = new int[2];
    delete[] p;
    // CHECK-NEXT: PreCall (operator new[]) [CXXAllocatorCall]
    // CHECK-NEXT: PostCall (operator new[]) [CXXAllocatorCall]
    // CHECK-NEXT: PreStmt<CXXNewExpr>
    // CHECK-NEXT: PostStmt<CXXNewExpr>
    // CHECK-NEXT: PreStmt<CXXDeleteExpr>
    // CHECK-NEXT: PostStmt<CXXDeleteExpr>
    // CHECK-NEXT: PreCall (operator delete[]) [CXXDeallocatorCall]
    // CHECK-NEXT: PostCall (operator delete[]) [CXXDeallocatorCall]

    p = new int[2];
    operator delete[](p, 23542368);
    // CHECK-NEXT: PreCall (operator new[]) [CXXAllocatorCall]
    // CHECK-NEXT: PostCall (operator new[]) [CXXAllocatorCall]
    // CHECK-NEXT: PreStmt<CXXNewExpr>
    // CHECK-NEXT: PostStmt<CXXNewExpr>
    // CHECK-NEXT: PreCall (operator delete[]) [SimpleFunctionCall]
    // CHECK-NEXT: PostCall (operator delete[]) [SimpleFunctionCall]

    void *v = operator new[](sizeof(int[2]), std::align_val_t(2));
    operator delete[](v, std::align_val_t(2));
    // CHECK-NEXT: PreCall (operator new[]) [SimpleFunctionCall]
    // CHECK-NEXT: PostCall (operator new[]) [SimpleFunctionCall]
    // CHECK-NEXT: PreCall (operator delete[]) [SimpleFunctionCall]
    // CHECK-NEXT: PostCall (operator delete[]) [SimpleFunctionCall]

    v = operator new[](sizeof(int[2]), std::align_val_t(2));
    operator delete[](v, 345345, std::align_val_t(2));
    // CHECK-NEXT: PreCall (operator new[]) [SimpleFunctionCall]
    // CHECK-NEXT: PostCall (operator new[]) [SimpleFunctionCall]
    // CHECK-NEXT: PreCall (operator delete[]) [SimpleFunctionCall]
    // CHECK-NEXT: PostCall (operator delete[]) [SimpleFunctionCall]

    p = new (std::nothrow) int[2];
    operator delete[](p, std::nothrow);
    // CHECK-NEXT: PreCall (operator new[]) [CXXAllocatorCall]
    // CHECK-NEXT: PostCall (operator new[]) [CXXAllocatorCall]
    // CHECK-NEXT: PreStmt<CXXNewExpr>
    // CHECK-NEXT: PostStmt<CXXNewExpr>
    // CHECK-NEXT: PreCall (operator delete[]) [SimpleFunctionCall]
    // CHECK-NEXT: PostCall (operator delete[]) [SimpleFunctionCall]

    v = operator new[](sizeof(int[2]), std::align_val_t(2), std::nothrow);
    operator delete[](v, std::align_val_t(2), std::nothrow);
    // CHECK-NEXT: PreCall (operator new[]) [SimpleFunctionCall]
    // CHECK-NEXT: PostCall (operator new[]) [SimpleFunctionCall]
    // CHECK-NEXT: PreCall (operator delete[]) [SimpleFunctionCall]
    // CHECK-NEXT: PostCall (operator delete[]) [SimpleFunctionCall]
  }
}
