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
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache(ArrayPtr[T::value + I:I + 5], T::array[(i + T::value, 5): 6])
  }
  for (int i = 0; i < 10; ++i) {
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache(NS::NSArray[NS::NSInt])
  }

  for (int i = 0; i < 10; ++i) {
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache(NS::NSArray[NS::NSInt : NS::NSInt])
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+2{{use of undeclared identifier 'NSArray'; did you mean 'NS::NSArray'}}
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache(NSArray[NS::NSInt : NS::NSInt])
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+2{{use of undeclared identifier 'NSInt'; did you mean 'NS::NSInt'}}
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache(NS::NSArray[NSInt : NS::NSInt])
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+2{{use of undeclared identifier 'NSInt'; did you mean 'NS::NSInt'}}
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache(NS::NSArray[NS::NSInt : NSInt])
  }
}

struct S {
  static constexpr int value = 5;
  static constexpr char array[] ={1,2,3,4,5};
};

void use() {
  func<S, 5>();
}
