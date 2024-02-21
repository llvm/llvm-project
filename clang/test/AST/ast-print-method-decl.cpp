// RUN: %clang_cc1 -ast-print -triple i386-linux-gnu %s -o - -std=c++20 | FileCheck %s

// CHECK: struct DelegatingCtor1 {
struct DelegatingCtor1 {
  // CHECK-NEXT: DelegatingCtor1();
  DelegatingCtor1();

  // CHECK-NEXT: DelegatingCtor1(int) : DelegatingCtor1() {
  DelegatingCtor1(int) : DelegatingCtor1() {
    // CHECK-NEXT: }
  }

  // CHECK-NEXT: };
};


// CHECK: struct DelegatingCtor2 {
struct DelegatingCtor2 {
  // CHECK-NEXT: template <typename Ty> DelegatingCtor2(Ty);
  template <typename Ty> DelegatingCtor2(Ty);

  // FIXME: Implicitly specialized method should not be output
  // CHECK-NEXT: template<> DelegatingCtor2<float>(float);

  // CHECK-NEXT: DelegatingCtor2(int X) : DelegatingCtor2((float)X) {
  DelegatingCtor2(int X) : DelegatingCtor2((float)X) {
  // CHECK-NEXT: }
  }

  // CHECK-NEXT: };
};

// CHECK: struct DelegatingCtor3 {
struct DelegatingCtor3 {
  // CHECK: DelegatingCtor3(auto);
  DelegatingCtor3(auto);

  // FIXME: Implicitly specialized method should not be output
  // CHECK: template<> DelegatingCtor3<const char *>(const char *);

  // CHECK: DelegatingCtor3(int) : DelegatingCtor3("") {
  DelegatingCtor3(int) : DelegatingCtor3("") {
  // CHECK-NEXT: }
  }

  // CHECK-NEXT: };
};

// CHECK: struct CurlyCtorInit {
struct CurlyCtorInit {
  // CHECK-NEXT: struct A {
  struct A {
    // CHECK-NEXT: int x;
    int x;
  // CHECK-NEXT: };
  };

  // CHECK-NEXT: A a;
  A a;
  // CHECK-NEXT: int i;
  int i;

  // FIXME: /*implicit*/(int)0 should not be output
  // CHECK-NEXT: CurlyCtorInit(int *) : a(), i(/*implicit*/(int)0) {
  CurlyCtorInit(int *) : a(), i() {
  // CHECK-NEXT: }
  }

  // CHECK-NEXT: CurlyCtorInit(int **) : a{}, i{} {
  CurlyCtorInit(int **) : a{}, i{} {
  // CHECK-NEXT: }
  }

  // CHECK-NEXT: CurlyCtorInit(int ***) : a({}), i(0) {
  CurlyCtorInit(int ***) : a({}), i(0) {
  // CHECK-NEXT: }
  }

  // FIXME: Implicit this should not be output
  // CHECK-NEXT: CurlyCtorInit(int ****) : a({.x = 0}), i(this->a.x) {
  CurlyCtorInit(int ****) : a({.x = 0}), i(a.x) {
  // CHECK-NEXT: }
  }

  // CHECK-NEXT: };
};


// CHECK: struct DefMethodsWithoutBody {
struct DefMethodsWithoutBody {
  // CHECK-NEXT: DefMethodsWithoutBody() = delete;
  DefMethodsWithoutBody() = delete;

  // CHECK-NEXT: DefMethodsWithoutBody() = default;
  ~DefMethodsWithoutBody() = default;

  // CHECK-NEXT: __attribute__((alias("X"))) void m1();
  void m1() __attribute__((alias("X")));

  // CHECK-NEXT: };
};


// ---- Check that implict (non-written) constructor initializers are not output

struct ImplicitCtorInit1 {
  int a;
};

// CHECK: struct ImplicitCtorInit2 : ImplicitCtorInit1 {
struct ImplicitCtorInit2 : ImplicitCtorInit1 {

  // CHECK-NEXT: ImplicitCtorInit2(int *) {
  ImplicitCtorInit2(int *) {
  // CHECK-NEXT: }
  }

  // CHECK-NEXT: ImplicitCtorInit2(int **) : ImplicitCtorInit1() {
  ImplicitCtorInit2(int **) : ImplicitCtorInit1() {
  // CHECK-NEXT: }
  }

  // CHECK-NEXT: };
};
