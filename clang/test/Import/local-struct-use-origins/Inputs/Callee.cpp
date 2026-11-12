struct Bar {
  void bar(int _a, bool _b) {
    {
      struct S { int a; };
      S s = { _a };
    }
    {
      struct S { bool b; };
      S t = { _b };
    }
  };
};
