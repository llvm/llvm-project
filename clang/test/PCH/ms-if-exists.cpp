// RUN: %clang_cc1 -x c++ -fms-extensions -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -fms-extensions -fsyntax-only -include-pch %t %s -verify

// RUN: %clang_cc1 -x c++ -fms-extensions -emit-pch -fpch-instantiate-templates -o %t %s
// RUN: %clang_cc1 -x c++ -fms-extensions -fsyntax-only -include-pch %t %s -verify

#ifndef HEADER
#define HEADER
template<typename T>
void f(T t) {
  __if_exists(T::foo) {
    { }
    t.foo();
  }

  __if_not_exists(T::bar) {
    int *i = t;
    { }
  }
}
#else
struct HasFoo { 
  void foo();
};
struct HasBar { 
  void bar(int);
  void bar(float);
};

template void f(HasFoo); // expected-note{{in instantiation of function template specialization 'f<HasFoo>' requested here}}
                         // expected-error@17{{no viable conversion from 'HasFoo' to 'int *'}}
template void f(HasBar);
#endif
