// RUN: %clang_cc1 -std=c2x -verify -pedantic -Wno-comments %s

void test_basic_types(void) {
  auto undefined;     // expected-error {{declaration of variable 'undefined' with deduced type 'auto' requires an initializer}}
  auto auto_int = 4;
  auto auto_long = 4UL;
  signed auto a = 1L; // expected-error {{'auto' cannot be signed or unsigned}}

  _Static_assert(_Generic(auto_int, int : 1));
  _Static_assert(_Generic(auto_long, unsigned long : 1));
}

void test_complex_types(void) {
  _Complex auto i = 12.0; // expected-error {{'_Complex auto' is invalid}}
}

void test_gnu_extensions(void) {
  auto t = ({ // expected-warning {{use of GNU statement expression extension}}
    auto b = 12;
    b;
  });
  _Static_assert(_Generic(t, int : 1));
}

void test_sizeof_typeof(void) {
  auto auto_size = sizeof(auto);  // expected-error {{expected expression}}
  typeof(auto) tpof = 4;          // expected-error {{expected expression}}
}

void test_casts(void) {
  auto int_cast = (int)(4 + 3);
  auto double_cast = (double)(1 / 3);
  auto long_cast = (long)(4UL + 3UL);
  auto auto_cast = (auto)(4 + 3); // expected-error {{expected expression}}

  _Static_assert(_Generic(int_cast, int : 1));
  _Static_assert(_Generic(double_cast, double : 1));
  _Static_assert(_Generic(long_cast, long : 1));
}

void test_compound_literral(void) {
  auto int_cl = (int){13};
  auto double_cl = (double){2.5};
  auto array[] = { 1, 2, 3 }; // expected-error {{cannot use 'auto' with array in C}}

  auto auto_cl = (auto){13};  // expected-error {{expected expression}}

  _Static_assert(_Generic(int_cl, int : 1));
  _Static_assert(_Generic(double_cl, double : 1));
}

void test_array_pointers(void) {
  double array[3] = { 0 };
  auto a = array;
  auto b = &array;

  _Static_assert(_Generic(array, double * : 1));
  _Static_assert(_Generic(a, double * : 1));
  _Static_assert(_Generic(b, double (*)[3] : 1));
}

void test_typeof() {
  int typeof_target();
  auto result = (typeof(typeof_target())){12};

  _Static_assert(_Generic(result, int : 1));
}

void test_qualifiers(const int y) {
  const auto a = 12;
  auto b = y;
  static auto c = 1UL;
  int* pa = &a; // expected-warning {{initializing 'int *' with an expression of type 'const int *' discards qualifiers}}
  const int* pb = &b;
  int* pc = &c; // expected-warning {{incompatible pointer types initializing 'int *' with an expression of type 'unsigned long *'}}

  _Static_assert(_Generic(a, int : 1));
  _Static_assert(_Generic(b, int : 1));
  _Static_assert(_Generic(c, unsigned long : 1));
  _Static_assert(_Generic(pa, int * : 1));
  _Static_assert(_Generic(pb, const int * : 1));
  _Static_assert(_Generic(pc, int * : 1));
}

void test_strings(void) {
  auto str = "this is a string";
  auto str2[] = "this is a string";       // expected-warning {{type inference of a declaration other than a plain identifier with optional trailing attributes is a Clang extension}}
  auto (str3) = "this is a string";
  auto (((str4))) = "this is a string";

  _Static_assert(_Generic(str, char * : 1));
  _Static_assert(_Generic(str2, char * : 1));
  _Static_assert(_Generic(str3, char * : 1));
  _Static_assert(_Generic(str4, char * : 1));
}

void test_pointers(void) {
  auto a = 12;
  auto *ptr = &a;                         // expected-warning {{type inference of a declaration other than a plain identifier with optional trailing attributes is a Clang extension}}
  auto *str = "this is a string";         // expected-warning {{type inference of a declaration other than a plain identifier with optional trailing attributes is a Clang extension}}
  const auto *str2 = "this is a string";  // expected-warning {{type inference of a declaration other than a plain identifier with optional trailing attributes is a Clang extension}}
  auto *b = &a;                           // expected-warning {{type inference of a declaration other than a plain identifier with optional trailing attributes is a Clang extension}}
  *b = &a;                                // expected-error {{incompatible pointer to integer conversion assigning to 'int' from 'int *'; remove &}}
  auto nptr = nullptr;

  _Static_assert(_Generic(a, int : 1));
  _Static_assert(_Generic(ptr, int * : 1));
  _Static_assert(_Generic(str, char * : 1));
  _Static_assert(_Generic(str2, const char * : 1));
  _Static_assert(_Generic(b, int * : 1));
  _Static_assert(_Generic(nptr, typeof(nullptr) : 1));
}

void test_prototypes(void) {
  extern void foo(int a, int array[({ auto x = 12; x;})]);  // expected-warning {{use of GNU statement expression extension}}
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

    _Static_assert(_Generic(b, double : 1));
    _Static_assert(_Generic(a, double : 1));
  }
}

[[clang::overloadable]] auto test(auto x) { // expected-error {{'auto' not allowed in function prototype}} \
                                               expected-error {{'auto' not allowed in function return type}}
  return x;
}
