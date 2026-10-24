struct S {
  void m(int x);

  S();
  S(const S&);
  S &operator=(const S&);

  void doNotDeserialize();

  operator const char*();
  operator char*();
};

struct Trivial {
  void doNotDeserialize();
};