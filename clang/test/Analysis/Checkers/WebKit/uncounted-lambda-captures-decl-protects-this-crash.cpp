// RUN: %clang_analyze_cc1 -analyzer-checker=webkit.UncountedLambdaCapturesChecker -verify %s

struct Foo {
  int x;
  int y;
  Foo(int x, int y) : x(x) , y(y) { }
};

template <typename T>
struct Baz {
  void ref() const;
  void deref() const;
  Foo operator*();
  bool operator!();
};

inline Foo operator*(const Foo& a, const Foo& b);

Baz<Foo> someFunction();
template <typename CallbackType> void bar(CallbackType callback) {
  auto baz = someFunction();
  callback(baz);
}

struct Obj {
  void ref() const;
  void deref() const;

  void foo(Foo foo) {
    bar([this](auto baz) {
      // expected-warning@-1{{Captured raw-pointer 'this' to uncounted type is unsafe [webkit.UncountedLambdaCapturesChecker]}}
      bar([this, foo = *baz, foo2 = !baz](auto&&) {
        // expected-warning@-1{{Captured raw-pointer 'this' to uncounted type is unsafe [webkit.UncountedLambdaCapturesChecker]}}
        someFunction();
      });
    });
  }
};
