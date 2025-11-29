// RUN: %clang_cc1 -fsyntax-only -verify -std=c++14 -triple x86_64-unknown-unknown %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++14 -triple x86_64-unknown-unknown %s -fexperimental-new-constant-interpreter

constexpr bool test_constexpr_valid() {
  constexpr int arr[10] = {};
  __builtin_assume_dereferenceable(arr, 40);
  return true;
}
static_assert(test_constexpr_valid(), "");

constexpr bool test_constexpr_partial() {
  constexpr int arr[10] = {};
  __builtin_assume_dereferenceable(&arr[5], 20);
  return true;
}
static_assert(test_constexpr_partial(), "");

constexpr bool test_constexpr_nullptr() { // expected-error {{constexpr function never produces a constant expression}}
  __builtin_assume_dereferenceable(nullptr, 4); // expected-note 2{{read of dereferenced null pointer is not allowed in a constant expression}}
  return true;
}
static_assert(test_constexpr_nullptr(), ""); // expected-error {{not an integral constant expression}} expected-note {{in call to}}

constexpr bool test_constexpr_too_large() { // expected-error {{constexpr function never produces a constant expression}}
  constexpr int arr[10] = {};
  __builtin_assume_dereferenceable(arr, 100); // expected-note 2{{read of dereferenced one-past-the-end pointer is not allowed in a constant expression}}
  return true;
}
static_assert(test_constexpr_too_large(), ""); // expected-error {{not an integral constant expression}} expected-note {{in call to}}

constexpr bool test_single_var() {
  constexpr int single_var = 42;
  __builtin_assume_dereferenceable(&single_var, 4);
  return true;
}
static_assert(test_single_var(), "");

constexpr bool test_exact_boundary() {
  constexpr int arr[10] = {};
  __builtin_assume_dereferenceable(&arr[9], 4);
  return true;
}
static_assert(test_exact_boundary(), "");

constexpr bool test_one_over() { // expected-error {{constexpr function never produces a constant expression}}
  constexpr int arr[10] = {};
  __builtin_assume_dereferenceable(&arr[9], 5); // expected-note 2{{read of dereferenced one-past-the-end pointer is not allowed in a constant expression}}
  return true;
}
static_assert(test_one_over(), ""); // expected-error {{not an integral constant expression}} expected-note {{in call to}}

constexpr bool test_zero_size() {
  constexpr int arr[10] = {};
  __builtin_assume_dereferenceable(arr, 0);
  return true;
}
static_assert(test_zero_size(), "");

constexpr bool test_struct_member() {
  struct S {
    int x;
    int y;
  };
  constexpr S s = {1, 2};
  __builtin_assume_dereferenceable(&s.x, 4);
  return true;
}
static_assert(test_struct_member(), "");

constexpr bool test_range_valid() {
  constexpr int range_data[5] = {1, 2, 3, 4, 5};
  __builtin_assume_dereferenceable(range_data, 5 * sizeof(int));
  return range_data[0] == 1;
}
static_assert(test_range_valid(), "");

constexpr bool test_range_invalid() { // expected-error {{constexpr function never produces a constant expression}}
  constexpr int range_data[5] = {1, 2, 3, 4, 5};
  __builtin_assume_dereferenceable(range_data, 6 * sizeof(int)); // expected-note 2{{read of dereferenced one-past-the-end pointer is not allowed in a constant expression}}
  return true;
}
static_assert(test_range_invalid(), ""); // expected-error {{not an integral constant expression}} expected-note {{in call to}}

constexpr int arr1[10] = {};
constexpr int valid = (__builtin_assume_dereferenceable(arr1, 40), 12);

constexpr int invalid = (__builtin_assume_dereferenceable((int*)123, 4), 12); // expected-error {{constexpr variable 'invalid' must be initialized by a constant expression}} expected-note {{cast that performs the conversions of a reinterpret_cast is not allowed in a constant expression}}

constexpr int arr2[5] = {1, 2, 3, 4, 5};
constexpr int too_large = (__builtin_assume_dereferenceable(arr2, 6 * sizeof(int)), 12); // expected-error {{constexpr variable 'too_large' must be initialized by a constant expression}} expected-note {{read of dereferenced one-past-the-end pointer is not allowed in a constant expression}}

constexpr int null = (__builtin_assume_dereferenceable(nullptr, 4), 12); // expected-error {{constexpr variable 'null' must be initialized by a constant expression}} expected-note {{read of dereferenced null pointer is not allowed in a constant expression}}

int b = 10;
const int f = (__builtin_assume_dereferenceable((char*)&b + 1, 3), 12);
int a = f;

int c[10] = {};
const int g = (__builtin_assume_dereferenceable((unsigned char*)c + 5, 35), 42);
int d = g;

long long ll = 100;
const int h = (__builtin_assume_dereferenceable((void*)&ll, 8), 99);
int e = h;

struct Foo { int x; int y; int z; };
Foo foo = {1, 2, 3};
const int i = (__builtin_assume_dereferenceable((short*)&foo + 2, 8), 77);
int j = i;

double darr[10] = {};
const int k = (__builtin_assume_dereferenceable((int*)darr, 40), 55);
int l = k;
