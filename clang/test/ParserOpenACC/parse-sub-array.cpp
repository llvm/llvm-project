// RUN: %clang_cc1 %s -verify -fopenacc

void Func(int i, int j) {
  int array[5];
#pragma acc parallel private(array[:])
  while (true);
#pragma acc parallel private(array[i:])
  while (true);
#pragma acc parallel private(array[:j])
  while (true);
#pragma acc parallel private(array[i:j])
  while (true);
#pragma acc parallel private(array[1:2])
  while (true);

  // expected-error@+1{{expected unqualified-id}}
#pragma acc parallel private(array[::])
  while (true);
  // expected-error@+2{{expected ']'}}
  // expected-note@+1{{to match this '['}}
#pragma acc parallel private(array[1::])
  while (true);
  // expected-error@+2{{expected ']'}}
  // expected-note@+1{{to match this '['}}
#pragma acc parallel private(array[:2:])
  while (true);
  // expected-error@+3{{expected unqualified-id}}
  // expected-error@+2{{expected ']'}}
  // expected-note@+1{{to match this '['}}
#pragma acc parallel private(array[::3])
  while (true);
  // expected-error@+2{{expected ']'}}
  // expected-note@+1{{to match this '['}}
#pragma acc parallel private(array[1:2:3])
  while (true);
}

template<typename T, unsigned I, auto &IPtr>// #IPTR
void TemplFunc() {
  T array[I];
  T array2[2*I];
  T t; // #tDecl
#pragma acc parallel private(array[:])
  while (true);
#pragma acc parallel private(array[t:])
  while (true);
#pragma acc parallel private(array[I-1:])
  while (true);
#pragma acc parallel private(array[IPtr:])
  while (true);
#pragma acc parallel private(array[:t])
  while (true);
#pragma acc parallel private(array[:I])
  while (true);
#pragma acc parallel private(array[:IPtr])
  while (true);
#pragma acc parallel private(array[t:t])
  while (true);
#pragma acc parallel private(array2[I:I])
  while (true);
#pragma acc parallel private(array[IPtr:IPtr])
  while (true);

  // expected-error@+1{{expected unqualified-id}}
#pragma acc parallel private(array[::])
  while (true);
  // expected-error@+3{{'t' is not a class, namespace, or enumeration}}
  // expected-note@#tDecl{{'t' declared here}}
  // expected-error@+1{{expected unqualified-id}}
#pragma acc parallel private(array[t::])
  while (true);
  // expected-error@+2{{expected ']'}}
  // expected-note@+1{{to match this '['}}
#pragma acc parallel private(array[:I:])
  while (true);
  // expected-error@+2{{no member named 'IPtr' in the global namespace}}
  // expected-note@#IPTR{{'IPtr' declared here}}
#pragma acc parallel private(array[::IPtr])
  while (true);
  // expected-error@+2{{expected ']'}}
  // expected-note@+1{{to match this '['}}
#pragma acc parallel private(array[IPtr:I:t])
  while (true);
}

void use() {
  static constexpr int SomeI = 1;
  TemplFunc<int, 5, SomeI>();
}
