// RUN: %clang_cc1 %s -verify -fopenacc

template<typename T>
void Func() {
#pragma acc parallel
    typename T::type I; //#ILOC
#pragma acc serial
    typename T::type IS; //#ILOCSERIAL
#pragma acc kernels
    typename T::type IK; //#ILOCKERNELS
}

struct S {
  using type = int;
};

void use() {
  Func<S>();
  // expected-error@#ILOC{{type 'int' cannot be used prior to '::' because it has no members}}
  // expected-note@+3{{in instantiation of function template specialization 'Func<int>' requested here}}
  // expected-error@#ILOCSERIAL{{type 'int' cannot be used prior to '::' because it has no members}}
  // expected-error@#ILOCKERNELS{{type 'int' cannot be used prior to '::' because it has no members}}
  Func<int>();
}
