// RUN: %clang_cc1 -fsyntax-only -verify %s
template<typename T>
int &f0(T);

template<typename T>
float &f0(T*);

void test_f0(int i, int *ip) {
  int &ir = f0(i);
  float &fr = f0(ip);
}

template<typename T, typename U>
int &f1(T, U);

template<typename T>
float &f1(T, T);

void test_f1(int i, float f) {
  int &ir = f1(i, f);
  float &fr1 = f1(i, i);
  float &fr2 = f1(f, f);
}

template<typename T, typename U>
struct A { };

template<typename T>
int &f2(T);

template<typename T, typename U>
float &f2(A<T, U>);

template<typename T>
double &f2(A<T, T>);

void test_f2(int i, A<int, float> aif, A<int, int> aii) {
  int &ir = f2(i);
  float &fr = f2(aif);
  double &dr = f2(aii);
}

template<typename T, typename U>
int &f3(T*, U); // expected-note{{candidate}}

template<typename T, typename U>
float &f3(T, U*); // expected-note{{candidate}}

void test_f3(int i, int *ip, float *fp) {
  int &ir = f3(ip, i);
  float &fr = f3(i, fp);
  f3(ip, ip); // expected-error{{ambiguous}}
}

template<typename T>
int &f4(T&);

template<typename T>
float &f4(const T&);

void test_f4(int i, const int ic) {
  int &ir1 = f4(i);
  float &fr1 = f4(ic);
}

template<typename T, typename U>
int &f5(T&, const U&); // expected-note{{candidate}}

template<typename T, typename U>
float &f5(const T&, U&); // expected-note{{candidate}}

void test_f5(int i, const int ic) {
  f5(i, i); // expected-error{{ambiguous}}
}

template<typename T, typename U>
int &f6(T&, U&);

template<typename T, typename U>
float &f6(const T&, U&);

void test_f6(int i, const int ic) {
  int &ir = f6(i, i);
  float &fr = f6(ic, ic);
}

struct CrazyFun {
  template<typename T, typename U> operator A<T, U>();
  template<typename T> operator A<T, T>();
};

void fun(CrazyFun cf) {
  A<int, float> aif = cf;
  A<int, int> aii = cf;  
}
