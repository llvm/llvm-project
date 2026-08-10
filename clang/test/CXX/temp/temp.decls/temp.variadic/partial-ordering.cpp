// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// expected-no-diagnostics

// Various tests related to partial ordering of variadic templates.
template<typename ...Types> struct tuple;

template<typename Tuple> 
struct X1 {
  static const unsigned value = 0;
};

template<typename Head, typename ...Tail>
struct X1<tuple<Head, Tail...> > {
  static const unsigned value = 1;
};

template<typename Head, typename ...Tail>
struct X1<tuple<Head, Tail&...> > {
  static const unsigned value = 2;
};

template<typename Head, typename ...Tail>
struct X1<tuple<Head&, Tail&...> > {
  static const unsigned value = 3;
};

int check0[X1<tuple<>>::value == 0? 1 : -1];
int check1[X1<tuple<int>>::value == 2? 1 : -1];
int check2[X1<tuple<int, int>>::value == 1? 1 : -1];
int check3[X1<tuple<int, int&>>::value == 2? 1 : -1];
int check4[X1<tuple<int&, int&>>::value == 3? 1 : -1];

// Partial ordering of function templates.
template<typename T1, typename T2, typename ...Rest>
int &f0(T1, T2, Rest...);

template<typename T1, typename T2>
float &f0(T1, T2);

void test_f0() {
  int &ir1 = f0(1, 2.0, 'a');
  float &fr1 = f0(1, 2.0);
}

template<typename T1, typename T2, typename ...Rest>
int &f1(T1, T2, Rest...);

template<typename T1, typename T2>
float &f1(T1, T2, ...);

void test_f1() {
  int &ir1 = f1(1, 2.0, 'a');
}

template<typename T1, typename T2, typename ...Rest>
int &f2(T1, T2, Rest...);

float &f2(...);

void test_f2() {
  int &ir1 = f2(1, 2.0, 'a');
}
