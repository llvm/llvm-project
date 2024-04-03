// RUN: %clang_cc1 %s -verify -fopenacc

namespace NS {
  static char* NSArray;// expected-note{{declared here}}
  static int NSInt;// expected-note 2{{declared here}}
}
char *getArrayPtr();
template<typename T, int I>
void func() {
  char *ArrayPtr = getArrayPtr();
  for (int i = 0; i < 10; ++i) {
    // expected-warning@+1{{OpenACC construct 'cache' not yet implemented, pragma ignored}}
    #pragma acc cache(ArrayPtr[T::value + I:I + 5], T::array[(i + T::value, 5): 6])
  }
  for (int i = 0; i < 10; ++i) {
    // expected-warning@+1{{OpenACC construct 'cache' not yet implemented, pragma ignored}}
    #pragma acc cache(NS::NSArray[NS::NSInt])
  }

  for (int i = 0; i < 10; ++i) {
    // expected-warning@+1{{OpenACC construct 'cache' not yet implemented, pragma ignored}}
    #pragma acc cache(NS::NSArray[NS::NSInt : NS::NSInt])
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+2{{use of undeclared identifier 'NSArray'; did you mean 'NS::NSArray'}}
    // expected-warning@+1{{OpenACC construct 'cache' not yet implemented, pragma ignored}}
    #pragma acc cache(NSArray[NS::NSInt : NS::NSInt])
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+2{{use of undeclared identifier 'NSInt'; did you mean 'NS::NSInt'}}
    // expected-warning@+1{{OpenACC construct 'cache' not yet implemented, pragma ignored}}
    #pragma acc cache(NS::NSArray[NSInt : NS::NSInt])
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+2{{use of undeclared identifier 'NSInt'; did you mean 'NS::NSInt'}}
    // expected-warning@+1{{OpenACC construct 'cache' not yet implemented, pragma ignored}}
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
  for (int i = 0; i < 10; ++i) {
    // expected-warning@+1{{OpenACC construct 'cache' not yet implemented, pragma ignored}}
    #pragma acc cache(s.array[s.value])
  }
  HasMembersArray Arrs;
  for (int i = 0; i < 10; ++i) {
    // expected-warning@+1{{OpenACC construct 'cache' not yet implemented, pragma ignored}}
    #pragma acc cache(Arrs.MemArr[3].array[4])
  }
  for (int i = 0; i < 10; ++i) {
    // expected-warning@+1{{OpenACC construct 'cache' not yet implemented, pragma ignored}}
    #pragma acc cache(Arrs.MemArr[3].array[1:4])
  }
  for (int i = 0; i < 10; ++i) {
    // FIXME: Once we have a new array-section type to represent OpenACC as
    // well, change this error message.
    // expected-error@+2{{OpenMP array section is not allowed here}}
    // expected-warning@+1{{OpenACC construct 'cache' not yet implemented, pragma ignored}}
    #pragma acc cache(Arrs.MemArr[3:4].array[1:4])
  }
  for (int i = 0; i < 10; ++i) {
    // expected-error@+2{{OpenMP array section is not allowed here}}
    // expected-warning@+1{{OpenACC construct 'cache' not yet implemented, pragma ignored}}
    #pragma acc cache(Arrs.MemArr[3:4].array[4])
  }
  for (int i = 0; i < 10; ++i) {
    // expected-error@+3{{expected ']'}}
    // expected-note@+2{{to match this '['}}
    // expected-warning@+1{{OpenACC construct 'cache' not yet implemented, pragma ignored}}
    #pragma acc cache(Arrs.MemArr[3:4:].array[4])
  }
  for (int i = 0; i < 10; ++i) {
    // expected-error@+2{{expected expression}}
    // expected-warning@+1{{OpenACC construct 'cache' not yet implemented, pragma ignored}}
    #pragma acc cache(Arrs.MemArr[:].array[4])
  }
  for (int i = 0; i < 10; ++i) {
    // expected-error@+2{{expected unqualified-id}}
    // expected-warning@+1{{OpenACC construct 'cache' not yet implemented, pragma ignored}}
    #pragma acc cache(Arrs.MemArr[::].array[4])
  }
  for (int i = 0; i < 10; ++i) {
    // expected-error@+4{{expected expression}}
    // expected-error@+3{{expected ']'}}
    // expected-note@+2{{to match this '['}}
    // expected-warning@+1{{OpenACC construct 'cache' not yet implemented, pragma ignored}}
    #pragma acc cache(Arrs.MemArr[: :].array[4])
  }
  for (int i = 0; i < 10; ++i) {
    // expected-error@+2{{expected expression}}
    // expected-warning@+1{{OpenACC construct 'cache' not yet implemented, pragma ignored}}
    #pragma acc cache(Arrs.MemArr[3:].array[4])
  }
  func<S, 5>();
}

