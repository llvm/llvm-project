// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace N0 {
  template<typename T>
  void f0();

  template<typename T>
  int x0 = 0;

  template<typename T>
  class C0;
}
using namespace N0;

template<>
void f0<int>(); // expected-error {{no function template matches}}

template<>
int x0<int>;

template<>
class C0<int>;

namespace N1 {
  namespace N2 {
    template<typename T>
    void f2();

    template<typename T>
    int x2 = 0;

    template<typename T>
    class C2;
  }
  using namespace N2;
}

template<>
void N1::f2<int>(); // expected-error {{no function template matches}}

template<>
int N1::x2<int>;

template<>
class N1::C2<int>;

namespace N3 {
  namespace N4 {
    template<typename T>
    void f4();

    template<typename T>
    int x4 = 0;

    template<typename T>
    class C4;
  }
  using N4::f4;
  using N4::x4;
  using N4::C4;
}

template<>
void N3::f4<int>(); // expected-error {{no function template matches}}

template<>
int N3::x4<int>;

template<>
class N3::C4<int>;

inline namespace N5 {
  template<typename T>
  void f5();

  template<typename T>
  int x5 = 0;

  template<typename T>
  class C5;
}

template<>
void f5<int>();

template<>
int x5<int>;

template<>
class C5<int>;

namespace N6 {
  inline namespace N7 {
    template<typename T>
    void f7();

    template<typename T>
    int x7 = 0;

    template<typename T>
    class C7;
  }
}

template<>
void N6::f7<int>();

template<>
int N6::x7<int>;

template<>
class N6::C7<int>;
