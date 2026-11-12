// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++14 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

namespace std {
  inline namespace __1 {
    template<bool, class = void> struct enable_if {};
    template<class T> struct enable_if<true, T> { using type = T; };
    template<bool B, class T = void> using enable_if_t = typename enable_if<B, T>::type;
  }
}

namespace similar_to_user_code {
  // expected-note@+2 {{candidate template ignored: requirement 'sizeof(char) != 1' was not satisfied [with T = char]}}
  template<class T, class = std::enable_if_t<sizeof(T) != 1>>
  void f(T, short);

  // expected-note@+2 {{candidate template ignored: requirement 'sizeof(char) != 1' was not satisfied [with T = char]}}
  template<class T, std::enable_if_t<sizeof(T) != 1>* = nullptr>
  void f(T, int);

  // expected-note@+2 {{candidate template ignored: requirement 'sizeof(char) != 1' was not satisfied [with T = char]}}
  template<class T>
  std::enable_if_t<sizeof(T) != 1, void> f(T, long);

  void test() {
    f('x', 0); // expected-error{{no matching function}}
  }
}

namespace similar_to_libcxx_version_14 {
  template<bool, class = void> struct enable_if {};
  template<class T> struct enable_if<true, T> { using type = T; };
  template<bool B, class T = void> using __enable_if_t = typename enable_if<B, T>::type;

  // expected-note@+2 {{candidate template ignored: requirement 'sizeof(char) != 1' was not satisfied [with T = char]}}
  template<class T, class = __enable_if_t<sizeof(T) != 1>>
  void f(T, short);

  // expected-note@+2 {{candidate template ignored: requirement 'sizeof(char) != 1' was not satisfied [with T = char]}}
  template<class T, __enable_if_t<sizeof(T) != 1>* = nullptr>
  void f(T, int);

  // expected-note@+2 {{candidate template ignored: requirement 'sizeof(char) != 1' was not satisfied [with T = char]}}
  template<class T>
  __enable_if_t<sizeof(T) != 1, void> f(T, long);

  void test() {
    f('x', 0); // expected-error{{no matching function}}
  }
}

namespace similar_to_libcxx_version_13 {
  template<bool> struct _MetaBase {};
  template<> struct _MetaBase<true> { template<class R> using _EnableIfImpl = R; };
  template<bool B, class T = void> using _EnableIf = typename _MetaBase<B>::template _EnableIfImpl<T>;

  // expected-note@+2 {{no member named '_EnableIfImpl'}}
  template<class T, class = _EnableIf<sizeof(T) != 1>>
  void f(T, short);

  // expected-note@+2 {{no member named '_EnableIfImpl'}}
  template<class T, _EnableIf<sizeof(T) != 1>* = nullptr>
  void f(T, int);

  // expected-note@+2 {{no member named '_EnableIfImpl'}}
  template<class T>
  _EnableIf<sizeof(T) != 1, void> f(T, long);

  void test() {
    f('x', 0); // expected-error{{no matching function}}
  }
}

namespace not_all_names_are_magic {
  template<bool, class = void> struct enable_if {};
  template<class T> struct enable_if<true, T> { using type = T; };
  template<bool B, class T = void> using a_pony = typename enable_if<B, T>::type;

  // expected-note@-2 {{candidate template ignored: disabled by 'enable_if' [with T = char]}}
  template<class T, class = a_pony<sizeof(T) != 1>>
  void f(T, short);

  // expected-note@-6 {{candidate template ignored: disabled by 'enable_if' [with T = char]}}
  template<class T, a_pony<sizeof(T) != 1>* = nullptr>
  void f(T, int);

  // expected-note@-10 {{candidate template ignored: disabled by 'enable_if' [with T = char]}}
  template<class T>
  a_pony<sizeof(T) != 1, void> f(T, long);

  void test() {
    f('x', 0); // expected-error{{no matching function}}
  }
}
