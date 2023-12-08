// RUN: %clang_cc1 -triple x86_64-apple-darwin -O1 -disable-llvm-passes -emit-llvm -std=c++11 -o - %s | FileCheck %s

template<typename T>
struct X0 {
  void f(T &t) {
    t = 0;
  }
  
  void g(T &t);
  
  void h(T &t);
  
  static T static_var;
};

template<typename T>
inline void X0<T>::g(T & t) {
  t = 0;
}

template<typename T>
void X0<T>::h(T & t) {
  t = 0;
}

template<typename T>
T X0<T>::static_var = 0;

extern template struct X0<int*>;

int *&test(X0<int*> xi, int *ip) {
  // CHECK: define available_externally void @_ZN2X0IPiE1fERS0_
  xi.f(ip);
  // CHECK: define available_externally void @_ZN2X0IPiE1gERS0_
  xi.g(ip);
  // CHECK: declare void @_ZN2X0IPiE1hERS0_
  xi.h(ip);
  return X0<int*>::static_var;
}

template<typename T>
void f0(T& t) {
  t = 0;
}

template<typename T>
inline void f1(T& t) {
  t = 0;
}

extern template void f0<>(int *&);
extern template void f1<>(int *&);

void test_f0(int *ip, float *fp) {
  // CHECK: declare void @_Z2f0IPiEvRT_
  f0(ip);
  // CHECK: define linkonce_odr void @_Z2f0IPfEvRT_
  f0(fp);
}

void test_f1(int *ip, float *fp) {
  // CHECK: define available_externally void @_Z2f1IPiEvRT_
  f1(ip);
  // CHECK: define linkonce_odr void @_Z2f1IPfEvRT_
  f1(fp);
}
