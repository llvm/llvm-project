// RUN: %clang_cc1 %s -Wno-uninitialized -std=c++1z -fsyntax-only -verify

const extern int arr[];
constexpr auto p = arr; // ok
constexpr int f(int i) {return p[i];} // expected-note {{read of dereferenced one-past-the-end pointer}}

constexpr int arr[] {1, 2, 3};
constexpr auto p2 = arr + 2; // ok
constexpr int x = f(2); // ok
constexpr int y = f(3); // expected-error {{constant expression}}
// expected-note-re@-1 {{in call to 'f({{.*}})'}}

// FIXME: consider permitting this case
struct A {int m[];} a;
constexpr auto p3 = a.m; // expected-error {{constant expression}} expected-note {{without known bound}}
constexpr auto p4 = a.m + 1; // expected-error {{constant expression}} expected-note {{without known bound}}

void g(int i) {  // expected-note 2{{declared here}}
  int arr[i]; // expected-warning {{variable length arrays in C++ are a Clang extension}} \
                 expected-note {{function parameter 'i' with unknown value cannot be used in a constant expression}}
  constexpr auto *p = arr + 2; // expected-error {{constant expression}} expected-note {{without known bound}}

  // FIXME: Give a better diagnostic here. The issue is that computing
  // sizeof(*arr2) within the array indexing fails due to the VLA.
  int arr2[2][i]; // expected-warning 2{{variable length arrays in C++ are a Clang extension}} \
                     expected-note {{function parameter 'i' with unknown value cannot be used in a constant expression}}
  constexpr int m = ((void)arr2[2], 0); // expected-error {{constant expression}}
}
