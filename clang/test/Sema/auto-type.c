// RUN: %clang_cc1 %s -fsyntax-only -Wno-strict-prototypes -verify -pedantic -std=c11
// RUN: %clang_cc1 %s -fsyntax-only -Wno-strict-prototypes -verify -pedantic -std=c11 -fexperimental-new-constant-interpreter

__auto_type a = 5; // expected-warning {{'__auto_type' is a GNU extension}}
__extension__ __auto_type a1 = 5;
#pragma clang diagnostic ignored "-Wgnu-auto-type"
__auto_type b = 5.0;
__auto_type c = &b;
__auto_type d = (struct {int a;}) {5};
_Static_assert(__builtin_types_compatible_p(__typeof(a), int), "");
__auto_type e = e; // expected-error {{variable 'e' declared with deduced type '__auto_type' cannot appear in its own initializer}}

struct s { __auto_type a; }; // expected-error {{'__auto_type' not allowed in struct member}}

__auto_type f = 1, g = 1.0; // expected-error {{'__auto_type' deduced as 'int' in declaration of 'f' and deduced as 'double' in declaration of 'g'}}

__auto_type h() {} // expected-error {{'__auto_type' not allowed in function return type}}

int i() {
  struct bitfield { int field:2; };
  __auto_type j = (struct bitfield){1}.field; // expected-error {{cannot pass bit-field as __auto_type initializer in C}}

}

int k(l)
__auto_type l; // expected-error {{'__auto_type' not allowed in K&R-style function parameter}}
{}

void Issue53652(void) {
  // Ensure that qualifiers all work the same way as GCC.
  const __auto_type cat = a;
  const __auto_type pcat = &a;
  volatile __auto_type vat = a;
  volatile __auto_type pvat = &a;
  restrict __auto_type rat = &a;
  _Atomic __auto_type aat1 = a;
  _Atomic __auto_type paat = &a;

  // GCC does not accept this either, for the same reason.
  _Atomic(__auto_type) aat2 = a; // expected-error {{'__auto_type' not allowed here}} \
                                 // expected-error {{type specifier missing, defaults to 'int'}}

  // Ensure the types are what we expect them to be, regardless of order we
  // pass the types.
  _Static_assert(__builtin_types_compatible_p(__typeof(cat), const int), "");
  _Static_assert(__builtin_types_compatible_p(const int, __typeof(cat)), "");
  _Static_assert(__builtin_types_compatible_p(__typeof(pcat), int *const), "");
  _Static_assert(__builtin_types_compatible_p(int *const, __typeof(pcat)), "");
  _Static_assert(__builtin_types_compatible_p(__typeof(vat), volatile int), "");
  _Static_assert(__builtin_types_compatible_p(volatile int, __typeof(vat)), "");
  _Static_assert(__builtin_types_compatible_p(__typeof(pvat), int *volatile), "");
  _Static_assert(__builtin_types_compatible_p(int *volatile, __typeof(pvat)), "");
  _Static_assert(__builtin_types_compatible_p(__typeof(rat), int *restrict), "");
  _Static_assert(__builtin_types_compatible_p(int *restrict, __typeof(rat)), "");
  _Static_assert(__builtin_types_compatible_p(__typeof(aat1), _Atomic int), "");
  _Static_assert(__builtin_types_compatible_p(_Atomic int, __typeof(aat1)), "");
  _Static_assert(__builtin_types_compatible_p(__typeof(paat), _Atomic(int *)), "");
  _Static_assert(__builtin_types_compatible_p(_Atomic(int *), __typeof(paat)), "");

  // Ensure the types also work in generic selection expressions. Remember, the
  // type of the expression argument to _Generic is treated as-if it undergoes
  // lvalue to rvalue conversion, which drops qualifiers. We're making sure the
  // use of __auto_type doesn't impact that.
  (void)_Generic(cat, int : 0);
  (void)_Generic(pcat, int * : 0);
  (void)_Generic(vat, int : 0);
  (void)_Generic(pvat, int * : 0);
  (void)_Generic(rat, int * : 0);
  (void)_Generic(aat1, int : 0);
  (void)_Generic(paat, int * : 0);

  // Ensure that trying to merge two different __auto_type types does not
  // decide that they are both the same type when they're actually different,
  // and that we reject when the types are the same.
  __auto_type i = 12;
  __auto_type f = 1.2f;
  (void)_Generic(a, __typeof__(i) : 0, __typeof__(f) : 1);
  (void)_Generic(a,
                 __typeof__(i) : 0,   // expected-note {{compatible type 'typeof (i)' (aka 'int') specified here}}
                 __typeof__(a) : 1);  // expected-error {{type 'typeof (a)' (aka 'int') in generic association compatible with previously specified type 'typeof (i)' (aka 'int')}}
}

void Issue55702(void) {
  // A controlling expression which uses __auto_type should not be
  // automatically compatible with every association; we should be using the
  // canonical type for that comparison.
  void *ptr = 0;
  __auto_type v = ptr;
  (void)_Generic(v, long double : 0, double : 0, default : 1); // OK
  _Static_assert(_Generic(v, long double : 0, default : 1) == 1, "fail");
}
