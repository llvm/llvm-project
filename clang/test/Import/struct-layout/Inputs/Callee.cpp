struct S {
  int a;
};

struct Bar {
  void bar(int _a) {
    S s = { _a };
  };
};
