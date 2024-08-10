// RUN: %clang_cc1 -triple arm64-apple-ios -std=c++17 -Wno-vla -fsyntax-only -verify -fptrauth-intrinsics %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -std=c++17 -Wno-vla -fsyntax-only -verify -fptrauth-intrinsics %s

// RUN: not %clang_cc1 -triple arm64-apple-ios -std=c++17 -Wno-vla -fsyntax-only %s 2>&1 | FileCheck %s
// CHECK: this target does not support pointer authentication

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
  constexpr unsigned d0 = __builtin_ptrauth_type_discriminator(MemFnTy);
  static_assert(d0 == __builtin_ptrauth_string_discriminator("_ZTSM1SFivE"));
  static_assert(d0 == 60844);
  static_assert(__builtin_ptrauth_type_discriminator(int (S::*)()) == d0);
  static_assert(__builtin_ptrauth_type_discriminator(decltype(memFnPtr)) == d0);
  static_assert(__builtin_ptrauth_type_discriminator(decltype(memFnPtr2)) == d0);
  static_assert(__builtin_ptrauth_type_discriminator(decltype(&S::foo)) == d0);
  static_assert(dependentOperandDisc<decltype(&S::foo)>() == d0);

  constexpr unsigned d1 = __builtin_ptrauth_type_discriminator(void (S::*)(int));
  static_assert(__builtin_ptrauth_string_discriminator("_ZTSM1SFviE") == d1);
  static_assert(d1 == 39121);

  constexpr unsigned d2 = __builtin_ptrauth_type_discriminator(void (S::*)(float));
  static_assert(__builtin_ptrauth_string_discriminator("_ZTSM1SFvfE") == d2);
  static_assert(d2 == 52453);

  constexpr unsigned d3 = __builtin_ptrauth_type_discriminator(int (*())[s]);
  static_assert(__builtin_ptrauth_string_discriminator("FPE") == d3);
  static_assert(d3 == 34128);

  int f4(float);
  constexpr unsigned d4 = __builtin_ptrauth_type_discriminator(decltype(f4));
  static_assert(__builtin_ptrauth_type_discriminator(int (*)(float)) == d4);
  static_assert(__builtin_ptrauth_string_discriminator("FifE") == d4);
  static_assert(d4 == 48468);

  int f5(int);
  constexpr unsigned d5 = __builtin_ptrauth_type_discriminator(decltype(f5));
  static_assert(__builtin_ptrauth_type_discriminator(int (*)(int)) == d5);
  static_assert(__builtin_ptrauth_type_discriminator(short (*)(short)) == d5);
  static_assert(__builtin_ptrauth_type_discriminator(char (*)(char)) == d5);
  static_assert(__builtin_ptrauth_type_discriminator(long (*)(long)) == d5);
  static_assert(__builtin_ptrauth_type_discriminator(unsigned int (*)(unsigned int)) == d5);
  static_assert(__builtin_ptrauth_type_discriminator(int (&)(int)) == d5);
  static_assert(__builtin_ptrauth_string_discriminator("FiiE") == d5);
  static_assert(d5 == 2981);

  int t;
  int vmarray[s];
  (void)__builtin_ptrauth_type_discriminator(t); // expected-error {{unknown type name 't'}}
  (void)__builtin_ptrauth_type_discriminator(&t); // expected-error {{expected a type}}
  (void)__builtin_ptrauth_type_discriminator(decltype(vmarray)); // expected-error {{cannot pass undiscriminated type 'decltype(vmarray)' (aka 'int[s]')}}
  (void)__builtin_ptrauth_type_discriminator(int *); // expected-error {{cannot pass undiscriminated type 'int *' to '__builtin_ptrauth_type_discriminator'}}
  (void)__builtin_ptrauth_type_discriminator(); // expected-error {{expected a type}}
  (void)__builtin_ptrauth_type_discriminator(int (*)(int), int (*)(int));
  // expected-error@-1 {{expected ')'}}
  // expected-note@-2 {{to match this '('}}
}
