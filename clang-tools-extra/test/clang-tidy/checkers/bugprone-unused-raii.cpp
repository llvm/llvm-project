// RUN: %check_clang_tidy %s bugprone-unused-raii %t -- -- -fno-delayed-template-parsing

struct Foo {
  Foo();
  Foo(int);
  Foo(int, int);
  ~Foo();
};

struct Bar {
  Bar();
};

struct FooBar {
  FooBar();
  Foo f;
};

template <typename T>
void qux() {
  T(42);
}

template <typename T>
struct TFoo {
  TFoo(T);
  ~TFoo();
};

Foo f();

struct Ctor {
  Ctor(int);
  Ctor() {
    Ctor(0); // TODO: warn here.
  }
};

template <typename T>
void templ() {
  T();
}

template <typename T>
void neverInstantiated() {
  T();
}

struct CtorDefaultArg {
  CtorDefaultArg(int i = 0);
  ~CtorDefaultArg();
};

template <typename T>
struct TCtorDefaultArg {
  TCtorDefaultArg(T i = 0);
  ~TCtorDefaultArg();
};

struct CtorTwoDefaultArg {
  CtorTwoDefaultArg(int i = 0, bool b = false);
  ~CtorTwoDefaultArg();
};

template <typename T>
void templatetest() {
  TCtorDefaultArg<T>();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: object destroyed immediately after creation; did you mean to name the object?
  // CHECK-FIXES: TCtorDefaultArg<T> give_me_a_name;
  TCtorDefaultArg<T>{};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: object destroyed immediately after creation; did you mean to name the object?
  // CHECK-FIXES: TCtorDefaultArg<T> give_me_a_name;

  TCtorDefaultArg<T>(T{});
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: object destroyed immediately after creation; did you mean to name the object?
  // CHECK-FIXES: TCtorDefaultArg<T> give_me_a_name(T{});
  TCtorDefaultArg<T>{T{}};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: object destroyed immediately after creation; did you mean to name the object?
  // CHECK-FIXES: TCtorDefaultArg<T> give_me_a_name{T{}};

  int i = 0;
  (void)i;
}

void test() {
  Foo(42);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: object destroyed immediately after creation; did you mean to name the object?
// CHECK-FIXES: Foo give_me_a_name(42);
  Foo(23, 42);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: object destroyed immediately after creation; did you mean to name the object?
// CHECK-FIXES: Foo give_me_a_name(23, 42);
  Foo();
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: object destroyed immediately after creation; did you mean to name the object?
// CHECK-FIXES: Foo give_me_a_name;
  TFoo<int>(23);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: object destroyed immediately after creation; did you mean to name the object?
// CHECK-FIXES: TFoo<int> give_me_a_name(23);

  FooBar();
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: object destroyed immediately after creation; did you mean to name the object?
// CHECK-FIXES: FooBar give_me_a_name;

  Foo{42};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: object destroyed immediately after creation; did you mean to name the object?
  // CHECK-FIXES: Foo give_me_a_name{42};
  FooBar{};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: object destroyed immediately after creation; did you mean to name the object?
  // CHECK-FIXES: FooBar give_me_a_name;

  CtorDefaultArg();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: object destroyed immediately after creation; did you mean to name the object?
  // CHECK-FIXES: CtorDefaultArg give_me_a_name;

  CtorTwoDefaultArg();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: object destroyed immediately after creation; did you mean to name the object?
  // CHECK-FIXES: CtorTwoDefaultArg give_me_a_name;

  TCtorDefaultArg<int>();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: object destroyed immediately after creation; did you mean to name the object?
  // CHECK-FIXES: TCtorDefaultArg<int> give_me_a_name;

  TCtorDefaultArg<int>{};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: object destroyed immediately after creation; did you mean to name the object?
  // CHECK-FIXES: TCtorDefaultArg<int> give_me_a_name;

  templ<FooBar>();
  templ<Bar>();

  Bar();
  f();
  qux<Foo>();

#define M Foo();
  M

  {
    Foo();
  }
  Foo();
}
