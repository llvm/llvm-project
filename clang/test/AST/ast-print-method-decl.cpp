// RUN: %clang_cc1 -ast-print %s -o - -std=c++20 | FileCheck %s

// CHECK: struct A {
struct A {
  // CHECK-NEXT: A();
  A();

  // CHECK-NEXT: A(int) : A() {
  A(int) : A() {
    // CHECK-NEXT: }
  }

  // CHECK-NEXT: };
};


// CHECK: struct B {
struct B {
  // CHECK-NEXT: template <typename Ty> B(Ty);
  template <typename Ty> B(Ty);

  // FIXME: Implicitly specialized method should not be output
  // CHECK-NEXT: template<> B<float>(float);

  // CHECK-NEXT: B(int X) : B((float)X) {
  B(int X) : B((float)X) {
  // CHECK-NEXT: }
  }

  // CHECK-NEXT: };
};

// CHECK: struct C {
struct C {
  // FIXME: template <> should not be output
  // CHECK: template <> C(auto);
  C(auto);

  // FIXME: Implicitly specialized method should not be output
  // CHECK: template<> C<const char *>(const char *);

  // CHECK: C(int) : C("") {
  C(int) : C("") {
  // CHECK-NEXT: }
  }

  // CHECK-NEXT: };
};
