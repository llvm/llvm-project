// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s -Wno-unused
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s -Wno-unused -fms-compatibility -DMSVC

namespace PR8446_1 {
struct A {
  typedef int BASE_VALUE;
};

void g(int &y) {}

template <typename BASE_CLASS>
void f(int &rValue) {
#if MSVC
// expected-warning@+4 {{missing 'typename' prior to dependent type name 'BASE_CLASS::BASE_VALUE'}}
#else
  // expected-error@+2 {{expected expression}}
#endif
  return g((BASE_CLASS::BASE_VALUE &)rValue);
}

int main() {
  int x;
  f<A>(x);
  return 0;
}
} // namespace PR8446_1


namespace PR8446_2 {
struct site_symmetry_ops {};

template <class wt>
struct class_ {
  template <class A1>
  void def(A1 const &a1) {}
};

template <class A1, class A2>
struct init {
  init() {}
};

struct special_position_site_parameter {
  typedef char scatterer_type;
};

template <class wt>
struct valued_asu_parameter_heir_wrapper {
  static class_<wt> wrap(char const *name) {
    return class_<wt>();
  }
};

template <class wt>
struct special_position_wrapper {
  static void wrap(char const *name) {
    valued_asu_parameter_heir_wrapper<wt>::wrap(name)
#if MSVC
    // expected-warning@+4 {{missing 'typename' prior to dependent type name 'wt::scatterer_type'}}
#else
    // expected-error@+2 {{expected expression}}
#endif
        .def(init<site_symmetry_ops const &, wt::scatterer_type *>());
  }
};

void wrap_special_position() {
  special_position_wrapper<special_position_site_parameter>::wrap("special_position_site_parameter");
}
} // namespace PR8446_2

namespace PR8446_3 {
int g(int);
template <typename T>
int f1(int x) {
  return g((T::InnerName & x) & x);
}

template <typename T>
int f2(int x) {
  return g((T::InnerName & 3) & x);
}

template <typename T>
int f3(int x) {
  return g((T::InnerName & (3)));
}

template <typename T>
int f4(int x) {
  return g((T::InnerName * 3) & x);
}
struct A {
  static const int InnerName = 42;
};
int main() {
  f1<A>(0);
  f2<A>(0);
  f3<A>(0);
  return f4<A>(0);
}
} // namespace PR8446_3
