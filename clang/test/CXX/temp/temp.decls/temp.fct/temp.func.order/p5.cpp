// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
template<class T> int &f(T); 
template<class T> float &f(T*, int=1); 

template<class T> int &g(T); 
template<class T> float &g(T*, ...);

int main() { 
  int* ip; 
  float &fr1 = f(ip); 
  float &fr2 = g(ip);
}
