// RUN: %clang_cc1 %s -verify -fopenacc

namespace NS {
  static char* NSArray; // expected-note {{'NS::NSArray' declared here}}
  static int NSInt;     // expected-note 2 {{'NS::NSInt' declared here}}
}
char *getArrayPtr();
template<typename T, int I>
void func() {
  char *ArrayPtr = getArrayPtr();
#pragma acc loop
  for (int i = 0; i < 10; ++i) {
    // expected-warning@+1{{left operand of comma operator has no effect}}
    #pragma acc cache(ArrayPtr[T::value + I:I + 3], T::array[(T::value, 2): 2])
  }
#pragma acc loop
  for (int i = 0; i < 10; ++i) {
    #pragma acc cache(NS::NSArray[NS::NSInt])
  }

#pragma acc loop
  for (int i = 0; i < 10; ++i) {
    #pragma acc cache(NS::NSArray[NS::NSInt : NS::NSInt])
  }

#pragma acc loop
  for (int i = 0; i < 10; ++i) {
    // expected-error@+1{{use of undeclared identifier 'NSArray'}}
    #pragma acc cache(NSArray[NS::NSInt : NS::NSInt])
  }

#pragma acc loop
  for (int i = 0; i < 10; ++i) {
    // expected-error@+1{{use of undeclared identifier 'NSInt'}}
    #pragma acc cache(NS::NSArray[NSInt : NS::NSInt])
  }

#pragma acc loop
  for (int i = 0; i < 10; ++i) {
    // expected-error@+1{{use of undeclared identifier 'NSInt'}}
    #pragma acc cache(NS::NSArray[NS::NSInt : NSInt])
  }
}

struct S {
  static constexpr int value = 5;
  static constexpr char array[] ={1,2,3,4,5};
};

struct Members {
  int value = 5;
  char array[5] ={1,2,3,4,5};
};
struct HasMembersArray {
  Members MemArr[4];
};


void use() {

  Members s;
#pragma acc loop
  for (int i = 0; i < 10; ++i) {
    #pragma acc cache(s.array[s.value])
  }
  HasMembersArray Arrs;
#pragma acc loop
  for (int i = 0; i < 10; ++i) {
    #pragma acc cache(Arrs.MemArr[3].array[4])
  }
#pragma acc loop
  for (int i = 0; i < 10; ++i) {
    #pragma acc cache(Arrs.MemArr[3].array[1:4])
  }
#pragma acc loop
  for (int i = 0; i < 10; ++i) {
    // expected-error@+1{{OpenACC sub-array is not allowed here}}
    #pragma acc cache(Arrs.MemArr[2:1].array[1:4])
  }
#pragma acc parallel loop
  for (int i = 0; i < 10; ++i) {
    // expected-error@+1{{OpenACC sub-array is not allowed here}}
    #pragma acc cache(Arrs.MemArr[2:1].array[4])
  }
#pragma acc parallel loop
  for (int i = 0; i < 10; ++i) {
    // expected-error@+2{{expected ']'}}
    // expected-note@+1{{to match this '['}}
    #pragma acc cache(Arrs.MemArr[3:4:].array[4])
  }
#pragma acc parallel loop
  for (int i = 0; i < 10; ++i) {
    // expected-error@+1{{OpenACC sub-array is not allowed here}}
    #pragma acc cache(Arrs.MemArr[:].array[4])
  }
#pragma acc parallel loop
  for (int i = 0; i < 10; ++i) {
    // expected-error@+1{{expected unqualified-id}}
    #pragma acc cache(Arrs.MemArr[::].array[4])
  }
#pragma acc parallel loop
  for (int i = 0; i < 10; ++i) {
    // expected-error@+3{{expected expression}}
    // expected-error@+2{{expected ']'}}
    // expected-note@+1{{to match this '['}}
    #pragma acc cache(Arrs.MemArr[: :].array[4])
  }
#pragma acc parallel loop
  for (int i = 0; i < 10; ++i) {
    // expected-error@+1{{OpenACC sub-array is not allowed here}}
    #pragma acc cache(Arrs.MemArr[3:].array[4])
  }
  func<S, 5>(); // expected-note{{in instantiation of function template specialization}}
}

