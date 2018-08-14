struct S0 {
  S0();
  S0(const S0 &) noexcept(false);
  int a;
};

struct S {
  void m() {
    __block S0 x, y;
    ^{ (void)x; (void)y; };
  }
};
