// RUN: %clang_cc1 -std=c++20 -verify %s
// expected-no-diagnostics

namespace GH57945 {
  template<typename T>
    concept c = true;

  template<typename>
    auto f = []() requires c<void> {
    };

  void g() {
      f<int>();
  };
}

namespace GH57945_2 {
  template<typename>
    concept c = true;

  template<typename T>
    auto f = [](auto... args) requires c<T>  {
    };

  template <typename T>
  auto f2 = [](auto... args)
    requires (sizeof...(args) > 0)
  {};

  void g() {
      f<void>();
      f2<void>(5.0);
  }
}

namespace GH57958 {
  template<class> concept C = true;
  template<int> constexpr bool v = [](C auto) { return true; }(0);
  int _ = v<0>;
}
namespace GH57958_2 {
  template<class> concept C = true;
  template<int> constexpr bool v = [](C auto...) { return true; }(0);
  int _ = v<0>;
}

namespace GH57971 {
  template<typename>
    concept any = true;

  template<typename>
    auto f = [](any auto) {
    };

  using function_ptr = void(*)(int);
  function_ptr ptr = f<void>;
}
