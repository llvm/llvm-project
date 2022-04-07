// RUN: %clang_cc1 -no-opaque-pointers -triple=x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

template <typename T>
struct Foo {
private:
  T x;

public:
  Foo(T x) : x(x) {}
  ~Foo() {}

  T get() { return x; }
  void set(T _x) { x = _x; }
};

template <typename T>
struct Bar {
private:
  struct Foo<T> foo;

public:
  Bar(T x) : foo(x) {}
  ~Bar() {}

  T get() { return foo.get(); }
  void set(T _x) { foo.set(_x); }
};

template <typename T>
struct Baz : Foo<T> {
public:
  Baz(T x) : Foo<T>(x) {}
  ~Baz() {}
};

// These two specializations should generate lines for all of Foo's methods.

template struct Foo<char>;

template struct Foo<short>;

// This should not generate lines for the implicit specialization of Foo, but
// should generate lines for the explicit specialization of Bar.

template struct Bar<int>;

// This should not generate lines for the implicit specialization of Foo, but
// should generate lines for the explicit specialization of Baz.

template struct Baz<long>;
