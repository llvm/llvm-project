// RUN: %clang_cc1 -triple arm64-apple-ios -std=c++17 -fsyntax-only -verify -fptrauth-intrinsics %s

struct S {
  virtual int foo();
};

template <class T>
constexpr unsigned dependentOperandDisc() {
  return __builtin_ptrauth_type_discriminator(T);
}

void test_builtin_ptrauth_type_discriminator(unsigned s) {
  typedef int (S::*MemFnTy)();
  MemFnTy memFnPtr;
  int (S::*memFnPtr2)();

  constexpr unsigned d = __builtin_ptrauth_type_discriminator(MemFnTy);
  static_assert(d == 60844);
  static_assert(__builtin_ptrauth_type_discriminator(int (S::*)()) == d);
  static_assert(__builtin_ptrauth_type_discriminator(decltype(memFnPtr)) == d);
  static_assert(__builtin_ptrauth_type_discriminator(decltype(memFnPtr2)) == d);
  static_assert(__builtin_ptrauth_type_discriminator(decltype(&S::foo)) == d);
  static_assert(dependentOperandDisc<decltype(&S::foo)>() == d);
  static_assert(__builtin_ptrauth_type_discriminator(void (S::*)(int)) == 39121);
  static_assert(__builtin_ptrauth_type_discriminator(void (S::*)(float)) == 52453);
  static_assert(__builtin_ptrauth_type_discriminator(int *) == 42396);

  int t;
  int vmarray[s];
  __builtin_ptrauth_type_discriminator(t); // expected-error {{unknown type name 't'}}
  __builtin_ptrauth_type_discriminator(&t); // expected-error {{expected a type}}
  __builtin_ptrauth_type_discriminator(decltype(vmarray)); // expected-error {{cannot pass variably-modified type 'decltype(vmarray)'}}
}
