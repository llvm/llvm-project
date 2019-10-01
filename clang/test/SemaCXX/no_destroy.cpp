// RUN: %clang_cc1 -DNO_DTORS -DNO_EXCEPTIONS -fno-c++-static-destructors -verify %s
// RUN: %clang_cc1 -DNO_EXCEPTIONS -verify %s
// RUN: %clang_cc1 -DNO_DTORS -fexceptions -fno-c++-static-destructors -verify %s
// RUN: %clang_cc1 -fexceptions -verify %s

struct SecretDestructor {
#ifndef NO_DTORS
  // expected-note@+2 4 {{private}}
#endif
private: ~SecretDestructor(); // expected-note + {{private}}
};

SecretDestructor sd1;
thread_local SecretDestructor sd2;
void locals() {
  static SecretDestructor sd3;
  thread_local SecretDestructor sd4;
}

#ifndef NO_DTORS
// SecretDestructor sd1;                  // expected-error@-8 {{private}}
// thread_local SecretDestructor sd2;     // expected-error@-8 {{private}}
// void locals() {
//   static SecretDestructor sd3;         // expected-error@-8 {{private}}
//   thread_local SecretDestructor sd4;   // expected-error@-8 {{private}}
// }
#endif

[[clang::always_destroy]] SecretDestructor sd6; // expected-error{{private}}
[[clang::always_destroy]] thread_local SecretDestructor sd7; // expected-error{{private}}

[[clang::no_destroy]] SecretDestructor sd8;

int main() {
  [[clang::no_destroy]] int p; // expected-error{{no_destroy attribute can only be applied to a variable with static or thread storage duration}}
  [[clang::always_destroy]] int p2; // expected-error{{always_destroy attribute can only be applied to a variable with static or thread storage duration}}
  [[clang::no_destroy]] static int p3;
  [[clang::always_destroy]] static int p4;
}

[[clang::always_destroy]] [[clang::no_destroy]] int p; // expected-error{{'no_destroy' and 'always_destroy' attributes are not compatible}} // expected-note{{here}}
[[clang::no_destroy]] [[clang::always_destroy]] int p2; // expected-error{{'always_destroy' and 'no_destroy' attributes are not compatible}} // expected-note{{here}}

[[clang::always_destroy]] void f() {} // expected-warning{{'always_destroy' attribute only applies to variables}}
struct [[clang::no_destroy]] DoesntApply {};  // expected-warning{{'no_destroy' attribute only applies to variables}}

[[clang::no_destroy(0)]] int no_args; // expected-error{{'no_destroy' attribute takes no arguments}}
[[clang::always_destroy(0)]] int no_args2; // expected-error{{'always_destroy' attribute takes no arguments}}

// expected-error@+1 {{temporary of type 'SecretDestructor' has private destructor}}
SecretDestructor arr[10];

void local_arrays() {
  // expected-error@+1 {{temporary of type 'SecretDestructor' has private destructor}}
  static SecretDestructor arr2[10];
  // expected-error@+1 {{temporary of type 'SecretDestructor' has private destructor}}
  thread_local SecretDestructor arr3[10];
}

struct Base {
  ~Base();
};
struct Derived1 {
  Derived1(int);
  Base b;
};
struct Derived2 {
  Derived1 b;
};

void dontcrash() {
  [[clang::no_destroy]] static Derived2 d2[] = {0, 0};
}

[[clang::no_destroy]] Derived2 d2[] = {0, 0};
