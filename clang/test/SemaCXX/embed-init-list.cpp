// RUN: %clang_cc1 -fsyntax-only -verify -Wno-c23-extensions %s
// expected-no-diagnostics

namespace std {
typedef decltype(sizeof(int)) size_t;

template <class _E> class initializer_list {
  const _E *__begin_;
  size_t __size_;

  constexpr initializer_list(const _E *__b, size_t __s)
      : __begin_(__b), __size_(__s) {}

public:
  constexpr initializer_list() : __begin_(nullptr), __size_(0) {}
};
} // namespace std

template <typename T> struct S {
  S(std::initializer_list<T>);
};

template <> struct S<char> {
  S(std::initializer_list<char>);
};

struct S1 {
  S<char> data;
  int a;
};

template <typename _Tp, std::size_t _Nm> void to_array(_Tp (&&__a)[_Nm]) {}


template<typename T>
void tfn(T) {}

void tests() {

  S<char>{{
#embed __FILE__
  }};

  S1 ss{std::initializer_list<char>{
#embed __FILE__
  }};

  S sss = {
#embed __FILE__
  };

  std::initializer_list<int> il{
#embed __FILE__
  };

  static constexpr auto initializer_list = std::initializer_list<char>{
#embed __FILE__
      , '\0'};

  static constexpr auto intinitializer_list = std::initializer_list<int>{
#embed __FILE__
      , '\0'};

  to_array({
#embed __FILE__
  });

  tfn<std::initializer_list<int>>({
#embed __FILE__
  });
}
