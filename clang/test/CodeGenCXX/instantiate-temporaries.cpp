// RUN: %clang_cc1 -triple x86_64-apple-macosx10.7.0 -emit-llvm -o - %s | FileCheck %s

struct X {
  X();
  ~X();
};

struct Y {
  X get();
};

struct X2 {
  X x;
};

template<typename T>
void call() {
  Y().get();
}

// CHECK-LABEL: define weak_odr void @_Z4callIiEvv
// CHECK: call void @_ZN1Y3getEv
// CHECK-NEXT: call void @_ZN1XD1Ev
// CHECK-NEXT: ret void
template void call<int>();  

template<typename T>
void compound_literal() {
  (X2){};
}

// CHECK-LABEL: define weak_odr void @_Z16compound_literalIiEvv
// CHECK: call void @_ZN1XC1Ev
// CHECK-NEXT: call void @_ZN2X2D1Ev
// CHECK-NEXT: ret void
template void compound_literal<int>();  

