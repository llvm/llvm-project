// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// expected-no-diagnostics

template<typename T>
struct classify_function {
  static const unsigned value = 0;
};

template<typename R, typename ...Args>
struct classify_function<R(Args...)> {
  static const unsigned value = 1;
};

template<typename R, typename ...Args>
struct classify_function<R(Args...) const> {
  static const unsigned value = 2;
};

template<typename R, typename ...Args>
struct classify_function<R(Args...) volatile> {
  static const unsigned value = 3;
};

template<typename R, typename ...Args>
struct classify_function<R(Args...) const volatile> {
  static const unsigned value = 4;
};

template<typename R, typename ...Args>
struct classify_function<R(Args......)> {
  static const unsigned value = 5;
};

template<typename R, typename ...Args>
struct classify_function<R(Args......) const> {
  static const unsigned value = 6;
};

template<typename R, typename ...Args>
struct classify_function<R(Args......) volatile> {
  static const unsigned value = 7;
};

template<typename R, typename ...Args>
struct classify_function<R(Args......) const volatile> {
  static const unsigned value = 8;
};

template<typename R, typename ...Args>
struct classify_function<R(Args......) &&> {
  static const unsigned value = 9;
};

template<typename R, typename ...Args>
struct classify_function<R(Args......) const &> {
  static const unsigned value = 10;
};

typedef void f0(int) const;
typedef void f1(int, float...) const volatile;
typedef void f2(int, double, ...) &&;
typedef void f3(int, double, ...) const &;

int check0[classify_function<f0>::value == 2? 1 : -1];
int check1[classify_function<f1>::value == 8? 1 : -1];
int check2[classify_function<f2>::value == 9? 1 : -1];
int check3[classify_function<f3>::value == 10? 1 : -1];
