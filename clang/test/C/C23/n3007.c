// RUN: %clang_cc1 -std=c2x -verify -pedantic -Wno-comments %s

/* WG14 N3007: Yes
 * Type Inference for object definitions
 */
void test_auto_int(void) {
  auto int auto_int = 12;
}

void test_qualifiers(int x, const int y, int * restrict z) {
  const auto a = x;
  auto b = y;
  static auto c = 1UL;
  int* pa = &a; // expected-warning {{initializing 'int *' with an expression of type 'const int *' discards qualifiers}}
  const int* pb = &b;
  int* pc = &c; // expected-warning {{incompatible pointer types initializing 'int *' with an expression of type 'unsigned long *'}}

  const int ci = 12;
  auto yup = ci;
  yup = 12;

  auto r_test = z;

  _Static_assert(_Generic(a, int : 1));
  _Static_assert(_Generic(c, unsigned long : 1));
  _Static_assert(_Generic(pa, int * : 1));
  _Static_assert(_Generic(pb, const int * : 1));
  _Static_assert(_Generic(r_test, int * : 1));
}

void test_atomic(void) {
  _Atomic auto i = 12;  // expected-error {{_Atomic cannot be applied to type 'auto' in C23}}
  _Atomic(auto) j = 12; // expected-error {{'auto' not allowed here}} \
                           expected-error {{a type specifier is required for all declarations}}

  _Atomic(int) foo(void);
  auto k = foo();

  _Static_assert(_Generic(&i, _Atomic auto *: 1)); // expected-error {{_Atomic cannot be applied to type 'auto' in C23}} \
                                                      expected-error {{'auto' not allowed here}}
  _Static_assert(_Generic(k, int: 1));
}

void test_double(void) {
  double A[3] = { 0 };
  auto pA = A;
  auto qA = &A;
  auto pi = 3.14;

  _Static_assert(_Generic(A, double * : 1));
  _Static_assert(_Generic(pA, double * : 1));
  _Static_assert(_Generic(qA, double (*)[3] : 1));
  _Static_assert(_Generic(pi, double : 1));
}

int test_auto_param(auto a) { // expected-error {{'auto' not allowed in function prototype}}
  return (int)(a * 2);
}

auto test_auto_return(float a, int b) { // expected-error {{'auto' not allowed in function return type}}
  return ((a * b) * (a / b));
}

[[clang::overloadable]] auto test(auto x) { // expected-error {{'auto' not allowed in function prototype}} \
                                               expected-error {{'auto' not allowed in function return type}}
  return x;
}

void test_sizeof_alignas(void) {
  (void)sizeof(auto);       // expected-error {{expected expression}}
  _Alignas(auto) int a[4];  // expected-error {{expected expression}}
}

void test_arrary(void) {
  auto a[4];          // expected-error {{'auto' not allowed in array declaration}}
  auto b[] = {1, 2};  // expected-error {{cannot use 'auto' with array in C}}
}

void test_initializer_list(void) {
  auto a = {};        // expected-error {{cannot use 'auto' with array in C}}
  auto b = { 0 };     // expected-error {{cannot use 'auto' with array in C}}
  auto c = { 1, };    // expected-error {{cannot use 'auto' with array in C}}
  auto d = { 1 , 2 }; // expected-error {{cannot use 'auto' with array in C}}
  auto e = (int [3]){ 1, 2, 3 };
}

void test_structs(void) {
  // FIXME: Both of these should be diagnosed as invalid underspecified
  // declarations as described in N3006.
  auto p1 = (struct { int a; } *)0;
  struct s;
  auto p2 = (struct s { int a; } *)0;

  struct B { auto b; };   // expected-error {{'auto' not allowed in struct member}}
}

void test_typedefs(void) {
  typedef auto auto_type;   // expected-error {{'auto' not allowed in typedef}}

  typedef auto (*fp)(void); // expected-error {{'auto' not allowed in typedef}}
  typedef void (*fp)(auto); // expected-error {{'auto' not allowed in function prototype}}

  _Generic(0, auto : 1);    // expected-error {{'auto' not allowed here}}
}

void test_misc(void) {
  auto something;                           // expected-error {{declaration of variable 'something' with deduced type 'auto' requires an initializer}}
  auto test_char = 'A';
  auto test_char_ptr = "test";
  auto test_char_ptr2[] = "another test";   // expected-warning {{type inference of a declaration other than a plain identifier with optional trailing attributes is a Clang extension}}
  auto auto_size = sizeof(auto);            // expected-error {{expected expression}}

  _Static_assert(_Generic(test_char, int : 1));
  _Static_assert(_Generic(test_char_ptr, char * : 1));
  _Static_assert(_Generic(test_char_ptr2, char * : 1));
}

void test_no_integer_promotions(void) {
  short s;
  auto a = s;
  _Generic(a, int : 1); // expected-error {{controlling expression type 'short' not compatible with any generic association type}}
}

void test_compound_literals(void) {
  auto a = (int){};
  auto b = (int){ 0 };
  auto c = (int){ 0, };
  auto d = (int){ 0, 1 };       // expected-warning {{excess elements in scalar initializer}}

  auto auto_cl = (auto){13};    // expected-error {{expected expression}}

  _Static_assert(_Generic(a, int : 1));
  _Static_assert(_Generic(b, int : 1));
  _Static_assert(_Generic(c, int : 1));
}

void test_pointers(void) {
  int a;
  auto *ptr = &a; // expected-warning {{type inference of a declaration other than a plain identifier with optional trailing attributes is a Clang extension}}
  auto *ptr2 = a; // expected-error {{variable 'ptr2' with type 'auto *' has incompatible initializer of type 'int'}} \
                     expected-warning {{type inference of a declaration other than a plain identifier with optional trailing attributes is a Clang extension}}
  auto nptr = nullptr;

  _Static_assert(_Generic(ptr, int * : 1));
  _Static_assert(_Generic(ptr2, int * : 1));
}

void test_scopes(void) {
  double a = 7;
  double b = 9;
  {
    auto a = a * a; // expected-error {{variable 'a' declared with deduced type 'auto' cannot appear in its own initializer}} \
                       expected-error {{variable 'a' declared with deduced type 'auto' cannot appear in its own initializer}}
  }
  {
    auto b = a * a;
    auto a = b;

    _Static_assert(_Generic(a, double : 1));
    _Static_assert(_Generic(b, double : 1));
  }
}

void test_loop(void) {
  auto j = 4;
  for (auto i = j; i < 2 * j; i++);

  _Static_assert(_Generic(j, int : 1));
}

#define AUTO_MACRO(_NAME, ARG, ARG2, ARG3) \
auto _NAME = ARG + (ARG2 / ARG3);

// This macro should only work with integers due to the usage of binary operators
#define AUTO_INT_MACRO(_NAME, ARG, ARG2, ARG3) \
auto _NAME = (ARG ^ ARG2) & ARG3;

void test_macros(int in_int) {
  auto a = in_int + 1;
  AUTO_MACRO(b, 1.3, 2.5f, 3);
  AUTO_INT_MACRO(c, 64, 23, 0xff);
  AUTO_INT_MACRO(not_valid, 51.5, 25, 0xff); // expected-error {{invalid operands to binary expression ('double' and 'int')}}

  auto result = (a + (int)b) - c;

  _Static_assert(_Generic(a, int : 1));
  _Static_assert(_Generic(b, double : 1));
  _Static_assert(_Generic(c, int : 1));
  _Static_assert(_Generic(result, int : 1));
}
