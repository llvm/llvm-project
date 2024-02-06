// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14.0 -verify -Wno-vla %s

// PR11925
int n;
int (&f())[n]; // expected-error {{function declaration cannot have variably modified type}}

namespace PR18581 {
  template<typename T> struct pod {};
  template<typename T> struct error {
    typename T::error e; // expected-error {{cannot be used prior to '::'}}
  };
  struct incomplete; // expected-note {{forward declaration}}

  void f(int n) {
    pod<int> a[n];
    error<int> b[n]; // expected-note {{instantiation}}
    incomplete c[n]; // expected-error {{incomplete}}
  }
}

void pr23151(int (&)[*]) { // expected-error {{variable length array must be bound in function definition}}
}

void test_fold() {
  char a1[(unsigned long)(int *)0+1]{}; // expected-warning{{variable length array folded to constant array as an extension}}
  char a2[(unsigned long)(int *)0+1] = {}; // expected-warning{{variable length array folded to constant array as an extension}}
  char a3[(unsigned long)(int *)0+1];
}

// Demonstrate that the check for a static_assert-like use of VLA does not
// crash when there's no array size expression at all.
void test_null_array_size_expr() {
  int array1[]; // expected-error {{definition of variable with array type needs an explicit size or an initializer}}
  int array2[] = { 1, 2, 3 };
}

// Show that the check for a static_assert-like use of a VLA properly handles a
// dependent array size expression.
template <typename Ty>
void func(int expr) {
  int array[sizeof(Ty) ? sizeof(Ty{}) : sizeof(int)];
  int old_style_assert[expr ? Ty::one : Ty::Neg_one]; // We don't diagnose as a VLA until instantiation
}
