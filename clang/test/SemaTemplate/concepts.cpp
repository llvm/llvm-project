// RUN: %clang_cc1 -std=c++20 -verify %s

namespace PR47043 {
  template<typename T> concept True = true;
  template<typename ...T> concept AllTrue1 = True<T>; // expected-error {{expression contains unexpanded parameter pack 'T'}}
  template<typename ...T> concept AllTrue2 = (True<T> && ...);
  static_assert(AllTrue2<int, float, char>);
}

namespace PR47025 {
  template<typename ...T> concept AllAddable1 = requires(T ...t) { (void(t + 1), ...); };
  template<typename ...T> concept AllAddable2 = (requires(T ...t) { (t + 1); } && ...); // expected-error {{requirement contains unexpanded parameter pack 't'}}
  template<typename ...T> concept AllAddable3 = (requires(T t) { (t + 1); } && ...);
  template<typename ...T> concept AllAddable4 = requires(T t) { (t + 1); }; // expected-error {{expression contains unexpanded parameter pack 'T'}}
  template<typename ...T> concept AllAddable5 = requires(T t) { (void(t + 1), ...); }; // expected-error {{does not contain any unexpanded}}
  template<typename ...T> concept AllAddable6 = (requires { (T() + 1); } && ...);
  template<typename ...T> concept AllAddable7 = requires { (T() + 1); }; // expected-error {{expression contains unexpanded parameter pack 'T'}}

  static_assert(AllAddable1<int, float>);
  static_assert(AllAddable3<int, float>);
  static_assert(AllAddable6<int, float>);
  static_assert(!AllAddable1<int, void>);
  static_assert(!AllAddable3<int, void>);
  static_assert(!AllAddable6<int, void>);
}
