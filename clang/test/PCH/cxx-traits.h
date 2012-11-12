// Header for PCH test cxx-traits.cpp

template<typename _Tp>
struct __is_pod {
  enum { __value };
};

template<typename _Tp>
struct __is_empty {
  enum { __value };
};

template<typename T, typename ...Args>
struct is_trivially_constructible {
  static const bool value = __is_trivially_constructible(T, Args...);
};
