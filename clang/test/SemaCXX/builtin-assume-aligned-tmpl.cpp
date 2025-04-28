// RUN: %clang_cc1 -fsyntax-only -verify %s

template<int z>
int test9(int *a) {
  a = (int *) __builtin_assume_aligned(a, z + 1); // expected-error {{requested alignment is not a power of 2}}
  return a[0];
}

void test9i(int *a) {
  test9<42>(a); // expected-note {{in instantiation of function template specialization 'test9<42>' requested here}}
}

template<typename T>
int test10(int *a, T z) {
  a = (int *) __builtin_assume_aligned(a, z + 1); // expected-error {{must be a constant integer}}
  return a[0];
}

int test10i(int *a) {
  return test10(a, 42); // expected-note {{in instantiation of function template specialization 'test10<int>' requested here}}
}

template <int q>
void *atest() __attribute__((assume_aligned(q))); // expected-error {{requested alignment is not a power of 2}}

template <int q, int o>
void *atest2() __attribute__((assume_aligned(q, o))); // expected-error {{requested alignment is not a power of 2}}

void test20() {
  atest<31>(); // expected-note {{in instantiation of function template specialization 'atest<31>' requested here}}
  atest<32>();

  atest2<31, 5>(); // expected-note {{in instantiation of function template specialization 'atest2<31, 5>' requested here}}
  atest2<32, 4>();
}

// expected-error@+1 {{invalid application of 'sizeof' to a function type}}
template<typename T> __attribute__((assume_aligned(sizeof(int(T()))))) T *f();
void test21() {
  void *p = f<void>(); // expected-note {{in instantiation of function template specialization 'f<void>' requested here}}
}

// expected-error@+1 {{functional-style cast from 'void' to 'int' is not allowed}}
template<typename T> __attribute__((assume_aligned(sizeof((int(T())))))) T *g();
void test23() {
  void *p = g<void>(); // expected-note {{in instantiation of function template specialization 'g<void>' requested here}}
}

template <typename T, int o>
T *atest3() __attribute__((assume_aligned(31, o))); // expected-error {{requested alignment is not a power of 2}}

template <typename T, int o>
T *atest4() __attribute__((assume_aligned(32, o)));

template<typename T>
T atest5(int) __attribute__((assume_aligned(2)));

// expected-warning@+1 {{'assume_aligned' attribute only applies to return values that are pointers or references}}
int atest6(int) __attribute__((assume_aligned(2)));

void test22() {
  atest3<int, 5>();
  atest4<int, 5>();
}

// expected-warning@+1 {{'assume_aligned' attribute only applies to Objective-C methods and functions}}
class __attribute__((assume_aligned(32))) x {
  int y;
};

// expected-warning@+1 {{'assume_aligned' attribute only applies to return values that are pointers or references}}
x foo() __attribute__((assume_aligned(32)));

struct s1 {
  static const int x = 32;
};

struct s2 {
  static const int x = 64;
};

struct s3 {
  static const int x = 63;
};

template <typename X>
void *atest5() __attribute__((assume_aligned(X::x))); // expected-error {{requested alignment is not a power of 2}}
void test24() {
  atest5<s1>();
  atest5<s2>();
  atest5<s3>(); // expected-note {{in instantiation of function template specialization 'atest5<s3>' requested here}}
}

