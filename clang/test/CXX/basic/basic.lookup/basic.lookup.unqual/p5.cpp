// RUN: %clang_cc1 -fsyntax-only -Wno-unused-result -verify=expected,pre-cxx20 %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -Wno-unused-result -verify=expected,since-cxx20 %s

// expected-no-diagnostics

namespace GH28181 {

struct S {
  using T = int;
  operator T() { return 42; }
};

void foo() {
  S{}.operator T();
}

}

namespace GH94052 {

namespace a {
template <typename> class b {
  public:
  typedef int string_type;
  operator string_type();
};
} // namespace a
template <class> void c() {
  (void)&a::b<char>::operator string_type;
}

}

namespace CXXScopeSpec {

struct S {
  template <class U>
  struct T {

  };

  template <class U>
  operator T<U>() { return 42; }
};

template <class U>
void foo() {
  &S::operator T<U>();
  S().operator T<U>();
}

}

namespace Regression {

namespace ns {
template <class T>
struct Bar {};
}

template <class T>
struct S {
  operator T();
};

template <class T>
void foo(T t) {
  t.operator ns::Bar<T>();
}

}
