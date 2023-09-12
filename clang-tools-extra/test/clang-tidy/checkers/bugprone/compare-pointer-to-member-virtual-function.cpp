// RUN: %check_clang_tidy %s bugprone-compare-pointer-to-member-virtual-function %t

struct A {
  virtual ~A();
  void f1();
  void f2();
  virtual void f3();
  virtual void f4();

  void g1(int);
};

bool Result;

void base() {
  Result = (&A::f1 == &A::f2);

  Result = (&A::f1 == &A::f3);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: A pointer to member virtual function shall only be tested for equality with null-pointer-constant.

  Result = (&A::f1 != &A::f3);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: A pointer to member virtual function shall only be tested for equality with null-pointer-constant.

  Result = (&A::f3 == nullptr);

  Result = (&A::f3 == &A::f4);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: A pointer to member virtual function shall only be tested for equality with null-pointer-constant.

  void (A::*V1)() = &A::f3;
  Result = (V1 == &A::f1);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: A pointer to member virtual function shall only be tested for equality with null-pointer-constant.
  // CHECK-MESSAGES: :7:3: note: potential member virtual function.
  // CHECK-MESSAGES: :8:3: note: potential member virtual function.
  Result = (&A::f1 == V1);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: A pointer to member virtual function shall only be tested for equality with null-pointer-constant.
  // CHECK-MESSAGES: :7:3: note: potential member virtual function.
  // CHECK-MESSAGES: :8:3: note: potential member virtual function.

  auto& V2 = V1;
  Result = (&A::f1 == V2);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: A pointer to member virtual function shall only be tested for equality with null-pointer-constant.
  // CHECK-MESSAGES: :7:3: note: potential member virtual function.
  // CHECK-MESSAGES: :8:3: note: potential member virtual function.

  void (A::*V3)(int) = &A::g1;
  Result = (V3 == &A::g1);
  Result = (&A::g1 == V3);
}

using B = A;
void usingRecordName() {
  Result = (&B::f1 == &B::f3);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: A pointer to member virtual function shall only be tested for equality with null-pointer-constant.

  void (B::*V1)() = &B::f1;
  Result = (V1 == &B::f1);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: A pointer to member virtual function shall only be tested for equality with null-pointer-constant.
  // CHECK-MESSAGES: :7:3: note: potential member virtual function.
  // CHECK-MESSAGES: :8:3: note: potential member virtual function.
}


typedef A C;
void typedefRecordName() {
  Result = (&C::f1 == &C::f3);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: A pointer to member virtual function shall only be tested for equality with null-pointer-constant.

  void (C::*V1)() = &C::f1;
  Result = (V1 == &C::f1);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: A pointer to member virtual function shall only be tested for equality with null-pointer-constant.
  // CHECK-MESSAGES: :7:3: note: potential member virtual function.
  // CHECK-MESSAGES: :8:3: note: potential member virtual function.
}
