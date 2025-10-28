// RUN: %clang_analyze_cc1 %s \
// RUN:  -analyzer-checker=debug.AnalysisOrder \
// RUN:  -analyzer-config debug.AnalysisOrder:EvalCall=true \
// RUN:  -analyzer-config debug.AnalysisOrder:PreCall=true \
// RUN:  -analyzer-config debug.AnalysisOrder:PostCall=true \
// RUN:  2>&1 | FileCheck %s

// This test ensures that eval::Call event will be triggered for constructors.

class C {
public:
  C(){};
  C(int x){};
  C(int x, int y){};
};

void foo() {
  C C0;
  C C1(42);
  C *C2 = new C{2, 3};
  delete C2;
}

// CHECK:  PreCall (C::C) [CXXConstructorCall]
// CHECK-NEXT:  EvalCall (C::C) {argno: 0} [CXXConstructorCall]
// CHECK-NEXT:  PostCall (C::C) [CXXConstructorCall]

// CHECK-NEXT:  PreCall (C::C) [CXXConstructorCall]
// CHECK-NEXT:  EvalCall (C::C) {argno: 1} [CXXConstructorCall]
// CHECK-NEXT:  PostCall (C::C) [CXXConstructorCall]

// CHECK-NEXT:  PreCall (operator new) [CXXAllocatorCall]
//    COMMENT: Operator new calls (CXXNewExpr) are intentionally not eval-called,
//    COMMENT: because it does not make sense to eval call user-provided functions.
//    COMMENT: 1) If the new operator can be inlined, then don't prevent it from
//    COMMENT:    inlining by having an eval-call of that operator.
//    COMMENT: 2) If it can't be inlined, then the default conservative modeling
//    COMMENT:    is what we anyways want anyway.
//    COMMENT: So the EvalCall event will not be triggered for operator new calls.
// CHECK-NOT:   EvalCall
// CHECK-NEXT:  PostCall (operator new) [CXXAllocatorCall]

// CHECK-NEXT:  PreCall (C::C) [CXXConstructorCall]
// CHECK-NEXT:  EvalCall (C::C) {argno: 2} [CXXConstructorCall]
// CHECK-NEXT:  PostCall (C::C) [CXXConstructorCall]

// CHECK-NEXT: PreCall (operator delete) [CXXDeallocatorCall]
//    COMMENT: Same reasoning as for CXXNewExprs above.
// CHECK-NOT:  EvalCall
// CHECK-NEXT: PostCall (operator delete) [CXXDeallocatorCall]
