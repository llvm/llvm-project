struct S {
  S(S&&);
  S(const S&);
};
struct Foo {
  Foo(const S &s);
  S s;
};
