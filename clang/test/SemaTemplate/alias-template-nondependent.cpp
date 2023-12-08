// RUN: %clang_cc1 -std=c++20 -verify %s

namespace PR46377 {
  template<typename> using IntPtr = int*;
  template<typename ...T> auto non_dependent_typedef() {
    typedef int(*P)(IntPtr<T>...);
    return P();
  }
  template<typename ...T> auto non_dependent_alias() {
    using P = int(*)(IntPtr<T>...);
    return P();
  }
  template<typename ...T> auto non_dependent_via_sizeof() {
    using P = int(*)(int(...pack)[sizeof(sizeof(T))]); // expected-error {{invalid application of 'sizeof'}}
    return P();
  }

  using a = int (*)(int*, int*);
  using a = decltype(non_dependent_typedef<void, void>());
  using a = decltype(non_dependent_alias<void, void>());
  using a = decltype(non_dependent_via_sizeof<float, float>());

  using b = decltype(non_dependent_via_sizeof<float, void>()); // expected-note {{instantiation of}}
}
