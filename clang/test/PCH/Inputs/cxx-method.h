struct S {
  void m(int x);

  S();
  S(const S&);

  operator const char*();
  operator char*();
};
