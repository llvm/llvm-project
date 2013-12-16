// Header for PCH test cxx-traits.cpp

template<typename _Tp>
struct __is_pod { // expected-warning {{keyword '__is_pod' will be treated as an identifier for the remainder of the translation unit}}
  enum { __value };
};

template<typename _Tp>
struct __is_empty { // expected-warning {{keyword '__is_empty' will be treated as an identifier for the remainder of the translation unit}}
  enum { __value };
};

template<typename T, typename ...Args>
struct is_trivially_constructible {
  static const bool value = __is_trivially_constructible(T, Args...);
};
