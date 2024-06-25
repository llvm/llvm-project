// RUN: %clang_cc1 %s -verify -fopenacc

struct Incomplete; // #INCOMPLETE
struct NotConvertible{} NC;

struct CorrectConvert {
  operator int();
} Convert;

constexpr int returns_3() { return 3; }

using FuncPtrTy = void (*)();
FuncPtrTy FuncPtrTyArray[2];

void Func(int i, int j) {
  int array[5];
  int VLA[i];
  int *ptr;
  void *void_ptr;

  // Follows int-expr rules, so only convertible to int.
  // expected-error@+1{{OpenACC sub-array bound requires expression of integer type ('struct NotConvertible' invalid}}
#pragma acc parallel private(array[NC:])
  while (true);

  // expected-error@+1{{OpenACC sub-array bound requires expression of integer type ('struct NotConvertible' invalid}}
#pragma acc parallel private(array[:NC])
  while (true);

  // expected-error@+2{{OpenACC sub-array bound requires expression of integer type ('struct NotConvertible' invalid}}
  // expected-error@+1{{OpenACC sub-array bound requires expression of integer type ('struct NotConvertible' invalid}}
#pragma acc parallel private(array[NC:NC])
  while (true);

  // expected-error@+2{{OpenACC sub-array bound requires expression of integer type ('struct NotConvertible' invalid}}
  // expected-error@+1{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is not an array}}
#pragma acc parallel private(ptr[NC:])
  while (true);

  // expected-error@+1{{OpenACC sub-array bound requires expression of integer type ('struct NotConvertible' invalid}}
#pragma acc parallel private(ptr[:NC])
  while (true);

  // expected-error@+2{{OpenACC sub-array bound requires expression of integer type ('struct NotConvertible' invalid}}
  // expected-error@+1{{OpenACC sub-array bound requires expression of integer type ('struct NotConvertible' invalid}}
#pragma acc parallel private(ptr[NC:NC])
  while (true);

  // These are convertible, so they work.
#pragma acc parallel private(array[Convert:Convert])
  while (true);

#pragma acc parallel private(ptr[Convert:Convert])
  while (true);


  // The length for "dynamically" allocated dimensions of an array must be
  // explicitly specified.

  // expected-error@+1{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is not an array}}
#pragma acc parallel private(ptr[3:])
  while (true);

  // expected-error@+1{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is an array of unknown bound}}
#pragma acc parallel private(VLA[3:])
  while (true);

#pragma acc parallel private(ptr[:3])
  while (true);

#pragma acc parallel private(VLA[:3])
  while (true);

  // Error if the length of the array + the initializer is bigger the the array
  // with known bounds.

  // expected-error@+1{{OpenACC sub-array length evaluated to a value (6) that would be out of the range of the subscripted array size of 5}}
#pragma acc parallel private(array[i:returns_3() + 3])
  while (true);

  // expected-error@+1{{OpenACC sub-array length evaluated to a value (6) that would be out of the range of the subscripted array size of 5}}
#pragma acc parallel private(array[:returns_3() + 3])
  while (true);

#pragma acc parallel private(array[:returns_3()])
  while (true);

  // expected-error@+1{{OpenACC sub-array specified range [3:3] would be out of the range of the subscripted array size of 5}}
#pragma acc parallel private(array[returns_3():returns_3()])
  while (true);

  // expected-error@+1{{OpenACC sub-array lower bound evaluated to a value (6) that would be out of the range of the subscripted array size of 5}}
#pragma acc parallel private(array[returns_3() + 3:])
  while (true);

  // expected-error@+1{{OpenACC sub-array lower bound evaluated to a value (6) that would be out of the range of the subscripted array size of 5}}
#pragma acc parallel private(array[returns_3() + 3:1])
  while (true);

  // Standard doesn't specify this, but negative values are likely not
  // permitted, so disallow them here until we come up with a good reason to do
  // otherwise.

  // expected-error@+1{{OpenACC sub-array lower bound evaluated to negative value -1}}
#pragma acc parallel private(array[returns_3() - 4 : ])
  while (true);

  // expected-error@+1{{OpenACC sub-array length evaluated to negative value -1}}
#pragma acc parallel private(array[: -1])
  while (true);

  Incomplete *IncompletePtr;
  // expected-error@+2{{OpenACC sub-array base is of incomplete type 'Incomplete'}}
  // expected-note@#INCOMPLETE{{forward declaration of 'Incomplete'}}
#pragma acc parallel private(IncompletePtr[0 :1])
  while (true);

  // expected-error@+1{{OpenACC sub-array base is of incomplete type 'void'}}
#pragma acc parallel private(void_ptr[0:1])
  while (true);

  // OK: these are function pointers.
#pragma acc parallel private(FuncPtrTyArray[0 :1])
  while (true);

  // expected-error@+1{{OpenACC sub-array cannot be of function type 'void ()'}}
#pragma acc parallel private(FuncPtrTyArray[0][0 :1])
  while (true);


  // expected-error@+1{{OpenACC sub-array subscripted value is not an array or pointer}}
#pragma acc parallel private(i[0:1])
  while (true);
}

template<typename T, typename U, typename V, unsigned I, auto &CEArray>
void Templ(int i){
  T array[I];
  T VLA[i];
  T *ptr;
  U NC;
  V Conv;

  // Convertible:
  // expected-error@+2{{OpenACC sub-array bound requires expression of integer type ('NotConvertible' invalid}}
  // expected-note@#INST{{in instantiation of function template specialization}}
#pragma acc parallel private(array[NC:])
  while (true);
  // expected-error@+1{{OpenACC sub-array bound requires expression of integer type ('NotConvertible' invalid}}
#pragma acc parallel private(array[:NC])
  while (true);

#pragma acc parallel private(array[Conv:])
  while (true);
#pragma acc parallel private(array[:Conv])
  while (true);

  // Need a length for unknown size.
  // expected-error@+1{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is not an array}}
#pragma acc parallel private(ptr[Conv:])
  while (true);
  // expected-error@+1{{OpenACC sub-array length is unspecified and cannot be inferred because the subscripted value is an array of unknown bound}}
#pragma acc parallel private(VLA[Conv:])
  while (true);
#pragma acc parallel private(ptr[:Conv])
  while (true);
#pragma acc parallel private(VLA[:Conv])
  while (true);

  // Out of bounds.
  // expected-error@+1{{OpenACC sub-array lower bound evaluated to a value (2) that would be out of the range of the subscripted array size of 2}}
#pragma acc parallel private(array[I:])
  while (true);

  // OK, don't know the value.
#pragma acc parallel private(array[i:])
  while (true);

  // expected-error@+1{{OpenACC sub-array length evaluated to a value (3) that would be out of the range of the subscripted array size of 2}}
#pragma acc parallel private(array[:I + 1])
  while (true);

  // expected-error@+1{{OpenACC sub-array lower bound evaluated to a value (5) that would be out of the range of the subscripted array size of 5}}
#pragma acc parallel private(CEArray[5:])
  while (true);

  // expected-error@+1{{OpenACC sub-array length evaluated to a value (6) that would be out of the range of the subscripted array size of 5}}
#pragma acc parallel private(CEArray[:2 + I + I])
  while (true);

  // expected-error@+1{{OpenACC sub-array length evaluated to a value (4294967295) that would be out of the range of the subscripted array size of 5}}
#pragma acc parallel private(CEArray[:1 - I])
  while (true);

  // expected-error@+1{{OpenACC sub-array lower bound evaluated to a value (4294967295) that would be out of the range of the subscripted array size of 5}}
#pragma acc parallel private(CEArray[1 - I:])
  while (true);

  T not_ptr;
  // expected-error@+1{{OpenACC sub-array subscripted value is not an array or pointer}}
#pragma acc parallel private(not_ptr[0:1])
  while (true);
}

void inst() {
  static constexpr int CEArray[5]={1,2,3,4,5};
  Templ<int, NotConvertible, CorrectConvert, 2, CEArray>(5); // #INST
}
