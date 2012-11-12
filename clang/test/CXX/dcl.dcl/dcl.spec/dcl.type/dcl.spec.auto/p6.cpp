// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++98 -Wno-c++11-extensions

template<typename T>
struct only {
  only(T);
  template<typename U> only(U) = delete;
};

namespace N
{
  auto a = "const char [16]", *p = &a;

  only<const char [16]> testA = a;
  only<const char **> testP = p;
}

void h() {
  auto b = 42ULL;
  only<unsigned long long> testB = b;

  for (auto c = 0; c < 100; ++c) {
    only<int> testC = c;
  }
}

void p3example() {
  auto x = 5;
  const auto *v = &x, u = 6;
  static auto y = 0.0;

  only<int> testX = x;
  only<const int*> testV = v;
  only<const int> testU = u;
  only<double> testY = y;
}

void f() {
  if (auto a = true) {
    only<bool> testA = a;
  }

  switch (auto a = 0) {
  case 0:
    only<int> testA = a;
  }

  while (auto a = false) {
    only<bool> testA = a;
  }

  for (; auto a = "test"; ) {
    only<const char[5]> testA = a;
  }

  auto *fail1 = 0; // expected-error {{variable 'fail1' with type 'auto *' has incompatible initializer of type 'int'}}
  int **p;
  const auto **fail2(p); // expected-error {{variable 'fail2' with type 'auto const **' has incompatible initializer of type 'int **'}}
}

struct S {
  void f();
  char g(int);
  float g(double);
  int m;

  void test() {
    auto p1 = &S::f;
    auto S::*p2 = &S::f;
    auto (S::*p3)() = &S::f;
    auto p4 = &S::g; // expected-error {{incompatible initializer of type '<overloaded function type>'}}
    auto S::*p5 = &S::g; // expected-error {{incompatible initializer of type '<overloaded function type>'}}
    auto (S::*p6)(int) = &S::g;
    auto p7 = &S::m;
    auto S::*p8 = &S::m;

    only<void (S::*)()> test1 = p1;
    only<void (S::*)()> test2 = p2;
    only<void (S::*)()> test3 = p3;
    only<char (S::*)(int)> test6 = p6;
    only<int (S::*)> test7 = p7;
    only<int (S::*)> test8 = p8;
  }
};

namespace PR10939 {
  struct X {
    int method(int);
    int method(float); 
  };

  template<typename T> T g(T);

  void f(X *x) {
    auto value = x->method; // expected-error {{reference to non-static member function must be called}}
    if (value) { }

    auto funcptr = &g<int>;
    int (*funcptr2)(int) = funcptr;
  }
}

// if the initializer is a braced-init-list, deduce auto as std::initializer_list<T>:
// see SemaCXX/cxx0x-initializer-stdinitializerlist.cpp
