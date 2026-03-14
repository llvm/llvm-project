struct S {
  S(S&&);
  S(const S&);
};
struct Foo {
  Foo(const S &s);
  // CHECK-FIXES: Foo(S s);
  S s;
};
