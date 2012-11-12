// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

class C {
  C(void*);
};

int f(const C&);
int f(unsigned long);

template<typename T> int f(const T* t) {
  return f(reinterpret_cast<unsigned long>(t));
}

