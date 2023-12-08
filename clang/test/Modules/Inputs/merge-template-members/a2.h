namespace N {
  template <typename> struct A {
    int n;
    A() : n() {}
  };

  // Create declaration of A<int>.
  typedef A<int> AI;
}
