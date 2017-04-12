template <typename T> struct A {
};

template <> struct A<int> {
  struct B {
    int f;
  };
};

template <> struct A<bool> {
  struct B {
    int g;
  };
};
