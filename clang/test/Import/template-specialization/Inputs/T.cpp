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


template <typename T> constexpr int f() { return 0; }
template <> constexpr int f<int>() { return 4; }
