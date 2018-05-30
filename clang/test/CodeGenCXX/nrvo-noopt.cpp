// RUN: %clang_cc1 -emit-llvm -O0 -o - %s | FileCheck %s

struct X {
  X();
  X(X&&);
};

// CHECK-LABEL: define void @_Z7test_00b
X test_00(bool b) {
  if (b) {
    // CHECK-NOT: call void @_ZN1XC1EOS_
    // CHECK: call void @_ZN1XC1Ev
    // CHECK-NEXT: br label %return
    X x;
    return x;
  } else {
    // CHECK-NOT: call void @_ZN1XC1EOS_
    // CHECK: call void @_ZN1XC1Ev
    // CHECK-NEXT: br label %return
    X x;
    return x;
  }
}

// CHECK-LABEL: define void @_Z7test_01b
X test_01(bool b) {
  if (b) {
    // CHECK-NOT: call void @_ZN1XC1EOS_
    // CHECK: call void @_ZN1XC1Ev
    // CHECK-NEXT: br label %return
    X x;
    return x;
  }
  // CHECK-NOT: call void @_ZN1XC1EOS_
  // CHECK: call void @_ZN1XC1Ev
  // CHECK-NEXT: br label %return
  X x;
  return x;
}

// CHECK-LABEL: define void @_Z7test_02b
X test_02(bool b) {
  // CHECK: call void @_ZN1XC1Ev
  X x;

  if (b) {
    // CHECK-NOT: call void @_ZN1XC1EOS_
    // CHECK: call void @_ZN1XC1Ev
    // CHECK-NEXT: br label %return
    X y;
    return y;
  }

  // CHECK-NOT: call void @_ZN1XC1Ev
  // CHECK: call void @_ZN1XC1EOS_
  // CHECK-NEXT: br label %return
  return x;
}
