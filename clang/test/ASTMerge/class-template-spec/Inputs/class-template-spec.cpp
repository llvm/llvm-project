namespace N0 {
  template<typename T>
  struct A {
    template<typename U>
    friend struct A;
  };

  template struct A<long>;
} // namespace N0

namespace N1 {
  template<typename T>
  struct A;

  template<typename T>
  struct A {
    template<typename U>
    friend struct A;
  };

  template struct A<long>;
} // namespace N1

namespace N2 {
  template<typename T>
  struct A {
    template<typename U>
    friend struct A;
  };

  template<typename T>
  struct A;

  template struct A<long>;
} // namespace N2

namespace N3 {
  struct A {
    template<typename T>
    friend struct B;
  };

  template<typename T>
  struct B { };

  template struct B<long>;
} // namespace N3
