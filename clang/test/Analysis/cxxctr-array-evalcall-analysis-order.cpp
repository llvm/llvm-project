// RUN: %clang_analyze_cc1 %s \
// RUN:  -analyzer-checker=debug.AnalysisOrder \
// RUN:  -analyzer-config debug.AnalysisOrder:PreCall=true \
// RUN:  -analyzer-config debug.AnalysisOrder:PostCall=true \
// RUN:  2>&1 | FileCheck %s

// This test ensures that eval::Call event will be triggered for constructors.

class C {
public:
  C(){};
};

void stack() {
  C arr[4];
  C *arr2 = new C[4];
  C arr3[2][2];
}

// C arr[4];
// CHECK:  PreCall (C::C) [CXXConstructorCall]
// CHECK-NEXT:  PostCall (C::C) [CXXConstructorCall]
// CHECK-NEXT:  PreCall (C::C) [CXXConstructorCall]
// CHECK-NEXT:  PostCall (C::C) [CXXConstructorCall]
// CHECK-NEXT:  PreCall (C::C) [CXXConstructorCall]
// CHECK-NEXT:  PostCall (C::C) [CXXConstructorCall]
// CHECK-NEXT:  PreCall (C::C) [CXXConstructorCall]
// CHECK-NEXT:  PostCall (C::C) [CXXConstructorCall]
//
// C *arr2 = new C[4];
// CHECK-NEXT:  PreCall (operator new[]) [CXXAllocatorCall]
// CHECK-NEXT:  PostCall (operator new[]) [CXXAllocatorCall]
// CHECK-NEXT:  PreCall (C::C) [CXXConstructorCall]
// CHECK-NEXT:  PostCall (C::C) [CXXConstructorCall]
// CHECK-NEXT:  PreCall (C::C) [CXXConstructorCall]
// CHECK-NEXT:  PostCall (C::C) [CXXConstructorCall]
// CHECK-NEXT:  PreCall (C::C) [CXXConstructorCall]
// CHECK-NEXT:  PostCall (C::C) [CXXConstructorCall]
// CHECK-NEXT:  PreCall (C::C) [CXXConstructorCall]
// CHECK-NEXT:  PostCall (C::C) [CXXConstructorCall]
//
// C arr3[2][2];
// CHECK-NEXT:  PreCall (C::C) [CXXConstructorCall]
// CHECK-NEXT:  PostCall (C::C) [CXXConstructorCall]
// CHECK-NEXT:  PreCall (C::C) [CXXConstructorCall]
// CHECK-NEXT:  PostCall (C::C) [CXXConstructorCall]
// CHECK-NEXT:  PreCall (C::C) [CXXConstructorCall]
// CHECK-NEXT:  PostCall (C::C) [CXXConstructorCall]
// CHECK-NEXT:  PreCall (C::C) [CXXConstructorCall]
// CHECK-NEXT:  PostCall (C::C) [CXXConstructorCall]
