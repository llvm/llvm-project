// RUN: %clang_cc1 %s -verify -fopenacc

template<typename T>
void Func() {
#pragma acc parallel
    typename T::type I; //#ILOC
}

struct S {
  using type = int;
};

void use() {
  Func<S>();
  // expected-error@#ILOC{{type 'int' cannot be used prior to '::' because it has no members}}
  // expected-note@+1{{in instantiation of function template specialization 'Func<int>' requested here}}
  Func<int>();
}
