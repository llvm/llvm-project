template<typename T> class Vector;

template<typename T> class List {
public:
  void push_back(T);
};

namespace A {
  class Y {
    template <typename T> friend class WhereAmI;
  };
}

template <typename T> class A::WhereAmI {
public:
  static void func() {}
};
