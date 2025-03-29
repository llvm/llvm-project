// RUN: %clang_cc1 -fc++-static-destructors=none -verify %s
// RUN: %clang_cc1 -fc++-static-destructors=thread-local -verify=expected,thread-local-dtors %s
// RUN: %clang_cc1 -verify=expected,thread-local-dtors,all-dtors %s
// RUN: %clang_cc1 -fexceptions -fc++-static-destructors=none -verify %s
// RUN: %clang_cc1 -fexceptions -fc++-static-destructors=thread-local -verify=expected,thread-local-dtors %s
// RUN: %clang_cc1 -fexceptions -verify=expected,thread-local-dtors,all-dtors %s

struct SecretDestructor {
private: ~SecretDestructor(); // expected-note + {{private}}
};

SecretDestructor sd1; // all-dtors-error{{private}}
thread_local SecretDestructor sd2; // thread-local-dtors-error{{private}}
void locals() {
  static SecretDestructor sd3; // all-dtors-error{{private}}
  thread_local SecretDestructor sd4; // thread-local-dtors-error{{private}}
}

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
