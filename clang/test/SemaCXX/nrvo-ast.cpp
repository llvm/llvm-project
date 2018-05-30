// RUN: %clang_cc1 -fcxx-exceptions -fsyntax-only -ast-dump -o - %s | FileCheck %s

struct X {
  X();
  X(const X&);
  X(X&&);
};

// CHECK-LABEL: FunctionDecl {{.*}} test_00
X test_00() {
  // CHECK: VarDecl {{.*}} x {{.*}} nrvo
  X x;
  return x;
}

// CHECK-LABEL: FunctionDecl {{.*}} test_01
X test_01(bool b) {
  // CHECK: VarDecl {{.*}} x {{.*}} nrvo
  X x;
  if (b)
    return x;
  return x;
}

// CHECK-LABEL: FunctionDecl {{.*}} test_02
X test_02(bool b) {
  // CHECK-NOT: VarDecl {{.*}} x {{.*}} nrvo
  X x;
  // CHECK-NOT: VarDecl {{.*}} y {{.*}} nrvo
  X y;
  if (b)
    return y;
  return x;
}

// CHECK-LABEL: FunctionDecl {{.*}} test_03
X test_03(bool b) {
  if (b) {
    // CHECK: VarDecl {{.*}} y {{.*}} nrvo
    X y;
    return y;
  }
  // CHECK: VarDecl {{.*}} x {{.*}} nrvo
  X x;
  return x;
}

extern "C" _Noreturn void exit(int) throw();

// CHECK-LABEL: FunctionDecl {{.*}} test_04
X test_04(bool b) {
  {
    // CHECK: VarDecl {{.*}} x {{.*}} nrvo
    X x;
    if (b)
      return x;
  }
  exit(1);
}

void may_throw();
// CHECK-LABEL: FunctionDecl {{.*}} test_05
X test_05() {
  try {
    may_throw();
    return X();
  } catch (X x) {
    // CHECK-NOT: VarDecl {{.*}} x {{.*}} nrvo
    return x;
  }
}

// CHECK-LABEL: FunctionDecl {{.*}} test_06
X test_06() {
  // CHECK-NOT: VarDecl {{.*}} x {{.*}} nrvo
  X x __attribute__((aligned(8)));
  return x;
}

// CHECK-LABEL: FunctionDecl {{.*}} test_07
X test_07(bool b) {
  if (b) {
    // CHECK: VarDecl {{.*}} x {{.*}} nrvo
    X x;
    return x;
  }
  return X();
}

// CHECK-LABEL: FunctionDecl {{.*}} test_08
X test_08(bool b) {
  if (b) {
    // CHECK: VarDecl {{.*}} x {{.*}} nrvo
    X x;
    return x;
  } else {
    // CHECK: VarDecl {{.*}} y {{.*}} nrvo
    X y;
    return y;
  }
}

template <typename T>
struct Y {
  Y();
  // CHECK-LABEL: CXXMethodDecl {{.*}} test_09 'Y<T> ()'
  // CHECK: VarDecl {{.*}} y 'Y<T>' nrvo

  // CHECK-LABEL: CXXMethodDecl {{.*}} test_09 'Y<int> ()'
  // CHECK: VarDecl {{.*}} y 'Y<int>' nrvo
  static Y test_09() {
    Y y;
    return y;
  }
};

struct Z {
  Z(const X&);
};

// CHECK-LABEL: FunctionDecl {{.*}} test_10 'A ()'
// CHECK: VarDecl {{.*}} b 'B' nrvo

// CHECK-LABEL: FunctionDecl {{.*}} test_10 'X ()'
// CHECK: VarDecl {{.*}} b {{.*}} nrvo

// CHECK-LABEL: FunctionDecl {{.*}} test_10 'Z ()'
// CHECK-NOT: VarDecl {{.*}} b {{.*}} nrvo
template <typename A, typename B>
A test_10() {
  B b;
  return b;
}

void instantiate() {
  Y<int>::test_09();
  test_10<X, X>();
  test_10<Z, X>();
}
