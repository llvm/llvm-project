// RUN: %check_clang_tidy %s zircon-temporary-objects %t -- \
// RUN:   -config="{CheckOptions: [{key: zircon-temporary-objects.Names, value: 'Foo;NS::Bar'}]}" \
// RUN:   -header-filter=.* \
// RUN: -- -std=c++11

// Should flag instances of Foo, NS::Bar.

class Foo {
public:
  Foo() = default;
  Foo(int Val) : Val(Val){};

private:
  int Val;
};

namespace NS {

class Bar {
public:
  Bar() = default;
  Bar(int Val) : Val(Val){};

private:
  int Val;
};

} // namespace NS

class Bar {
public:
  Bar() = default;
  Bar(int Val) : Val(Val){};

private:
  int Val;
};

int func(Foo F) { return 1; };

int main() {
  Foo F;
  Foo *F2 = new Foo();
  new Foo();
  Foo();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: creating a temporary object of type 'Foo' is prohibited
  Foo F3 = Foo();
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: creating a temporary object of type 'Foo' is prohibited

  Bar();
  NS::Bar();
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: creating a temporary object of type 'NS::Bar' is prohibited

  int A = func(Foo());
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: creating a temporary object of type 'Foo' is prohibited

  Foo F4(0);
  Foo *F5 = new Foo(0);
  new Foo(0);
  Foo(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: creating a temporary object of type 'Foo' is prohibited
  Foo F6 = Foo(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: creating a temporary object of type 'Foo' is prohibited

  Bar(0);
  NS::Bar(0);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: creating a temporary object of type 'NS::Bar' is prohibited

  int B = func(Foo(0));
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: creating a temporary object of type 'Foo' is prohibited
}

namespace NS {

void f() {
  Bar();
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: creating a temporary object of type 'NS::Bar' is prohibited
  Bar(0);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: creating a temporary object of type 'NS::Bar' is prohibited
}

} // namespace NS

template <typename Ty>
Ty make_ty() { return Ty(); }
// CHECK-MESSAGES: :[[@LINE-1]]:23: warning: creating a temporary object of type 'Foo' is prohibited
// CHECK-MESSAGES: :[[@LINE-2]]:23: warning: creating a temporary object of type 'NS::Bar' is prohibited

void ty_func() {
  make_ty<Bar>();
  make_ty<NS::Bar>();
  make_ty<Foo>();
}

// Inheriting the disallowed class does not trigger the check.

class Bingo : NS::Bar {}; // Not explicitly disallowed

void f2() {
  Bingo();
}

template <typename Ty>
class Quux : Ty {};

void f3() {
  Quux<NS::Bar>();
  Quux<Bar>();
}
