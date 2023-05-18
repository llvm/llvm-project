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
}

// CHECK:  PreCall (C::C) [CXXConstructorCall]
// CHECK-NEXT:  EvalCall (C::C) {argno: 0} [CXXConstructorCall]
// CHECK-NEXT:  PostCall (C::C) [CXXConstructorCall]
// CHECK-NEXT:  PreCall (C::C) [CXXConstructorCall]
// CHECK-NEXT:  EvalCall (C::C) {argno: 1} [CXXConstructorCall]
// CHECK-NEXT:  PostCall (C::C) [CXXConstructorCall]
// CHECK-NEXT:  PreCall (operator new) [CXXAllocatorCall]
// CHECK-NEXT:  PostCall (operator new) [CXXAllocatorCall]
// CHECK-NEXT:  PreCall (C::C) [CXXConstructorCall]
// CHECK-NEXT:  EvalCall (C::C) {argno: 2} [CXXConstructorCall]
// CHECK-NEXT:  PostCall (C::C) [CXXConstructorCall]
