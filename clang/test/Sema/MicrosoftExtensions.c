// RUN: %clang_cc1 -triple i686-windows %s -fsyntax-only -Wno-unused-value -Wno-pointer-to-int-cast -Wmicrosoft -verify -fms-extensions

struct A
{
   int a[];  /* expected-warning {{flexible array member 'a' in otherwise empty struct is a Microsoft extension}} */
};

struct PR28407
{
  int : 1;
  int a[]; /* expected-warning {{flexible array member 'a' in otherwise empty struct is a Microsoft extension}} */
};

struct C {
   int l;
   union {
       int c1[];   /* expected-warning {{flexible array member 'c1' in a union is a Microsoft extension}}  */
       char c2[];  /* expected-warning {{flexible array member 'c2' in a union is a Microsoft extension}} */
   };
};


struct D {
   int l;
   int D[];
};

struct __declspec(uuid("00000000-0000-0000-C000-000000000046")) IUnknown {}; /* expected-error {{'uuid' attribute is not supported in C}} */

[uuid("00000000-0000-0000-C000-000000000046")] struct IUnknown2 {}; /* expected-error {{'uuid' attribute is not supported in C}} */

typedef struct notnested {
  long bad1;
  long bad2;
} NOTNESTED;


typedef struct nested1 {
  long a;
  struct notnested var1;
  NOTNESTED var2;
} NESTED1;

struct nested2 {
  long b;
  NESTED1;  // expected-warning {{anonymous structs are a Microsoft extension}}
};

struct nested2 PR20573 = { .a = 3 };

struct nested3 {
  long d;
  struct nested4 { // expected-warning {{anonymous structs are a Microsoft extension}}
    long e;
  };
  union nested5 { // expected-warning {{anonymous unions are a Microsoft extension}}
    long f;
  };
};

typedef union nested6 {
  long f;
} NESTED6;

struct test {
  int c;
  struct nested2;   // expected-warning {{anonymous structs are a Microsoft extension}}
  NESTED6;   // expected-warning {{anonymous unions are a Microsoft extension}}
};

void foo()
{
  struct test var;
  var.a;
  var.b;
  var.c;
  var.bad1;   // expected-error {{no member named 'bad1' in 'struct test'}}
  var.bad2;   // expected-error {{no member named 'bad2' in 'struct test'}}
}

// Enumeration types with a fixed underlying type.
const int seventeen = 17;
typedef int Int;

struct X0 {
  enum E1 : Int { SomeOtherValue } field;  // expected-warning{{enumeration types with a fixed underlying type are a Microsoft extension}}
  enum E1 : seventeen;
};

enum : long long {  // expected-warning{{enumeration types with a fixed underlying type are a Microsoft extension}}
  SomeValue = 0x100000000
};

void pointer_to_integral_type_conv(char* ptr) {
  char ch = (char)ptr;
  short sh = (short)ptr;
  ch = (char)ptr;
  sh = (short)ptr;

  // This is valid ISO C.
  _Bool b = (_Bool)ptr;
}

typedef struct {
  UNKNOWN u; // expected-error {{unknown type name 'UNKNOWN'}}
} AA;

typedef struct {
  AA; // expected-warning {{anonymous structs are a Microsoft extension}}
} BB;

struct anon_fault {
  struct undefined; // expected-warning {{anonymous structs are a Microsoft extension}}
                    // expected-error@-1 {{field has incomplete type 'struct undefined'}}
                    // expected-note@-2 {{forward declaration of 'struct undefined'}}
};

const int anon_falt_size = sizeof(struct anon_fault);

__declspec(deprecated("This is deprecated")) enum DE1 { one, two } e1; // expected-note {{'e1' has been explicitly marked deprecated here}}
struct __declspec(deprecated) DS1 { int i; float f; }; // expected-note {{'DS1' has been explicitly marked deprecated here}}

#define MY_TEXT		"This is also deprecated"
__declspec(deprecated(MY_TEXT)) void Dfunc1( void ) {} // expected-note {{'Dfunc1' has been explicitly marked deprecated here}}

struct __declspec(deprecated(123)) DS2 {};	// expected-error {{'deprecated' attribute requires a string}}

void test( void ) {
	e1 = one;	// expected-warning {{'e1' is deprecated: This is deprecated}}
	struct DS1 s = { 0 };	// expected-warning {{'DS1' is deprecated}}
	Dfunc1();	// expected-warning {{'Dfunc1' is deprecated: This is also deprecated}}

	enum DE1 no;	// no warning because E1 is not deprecated
}

int __sptr wrong1; // expected-error {{'__sptr' attribute only applies to pointer arguments}}
// The modifier must follow the asterisk
int __sptr *wrong_psp; // expected-error {{'__sptr' attribute only applies to pointer arguments}}
int * __sptr __uptr wrong2; // expected-error {{'__sptr' and '__uptr' attributes are not compatible}}
int * __sptr __sptr wrong3; // expected-warning {{attribute '__sptr' is already applied}}

// It is illegal to overload based on the type attribute.
void ptr_func(int * __ptr32 i) {}  // expected-note {{previous definition is here}}
void ptr_func(int * __ptr64 i) {} // expected-error {{redefinition of 'ptr_func'}}

// It is also illegal to overload based on the pointer type attribute.
void ptr_func2(int * __sptr __ptr32 i) {}  // expected-note {{previous definition is here}}
void ptr_func2(int * __uptr __ptr32 i) {} // expected-error {{redefinition of 'ptr_func2'}}

// Check for warning when return types have the type attribute.
void *__ptr32 ptr_func3() { return 0; } // expected-note {{previous definition is here}}
void *__ptr64 ptr_func3() { return 0; } // expected-error {{redefinition of 'ptr_func3'}}

// Test that __ptr32/__ptr64 can be passed as arguments with other address
// spaces.
void ptr_func4(int *i);
void ptr_func5(int *__ptr32 i);
void test_ptr_arguments() {
  int *__ptr64 i64;
  ptr_func4(i64);
  ptr_func5(i64);
}

int * __sptr __ptr32 __sptr wrong4; // expected-warning {{attribute '__sptr' is already applied}}

__ptr32 int *wrong5; // expected-error {{'__ptr32' attribute only applies to pointer arguments}}

int *wrong6 __ptr32;  // expected-error {{expected ';' after top level declarator}} expected-warning {{declaration does not declare anything}}

int * __ptr32 __ptr64 wrong7;  // expected-error {{'__ptr32' and '__ptr64' attributes are not compatible}}

int * __ptr32 __ptr32 wrong8;	// expected-warning {{attribute '__ptr32' is already applied}}

int *(__ptr32 __sptr wrong9); // expected-error {{'__sptr' attribute only applies to pointer arguments}} // expected-error {{'__ptr32' attribute only applies to pointer arguments}}

typedef int *T;
T __ptr32 wrong10; // expected-error {{'__ptr32' attribute only applies to pointer arguments}}

typedef char *my_va_list;
void __va_start(my_va_list *ap, ...); // expected-note {{passing argument to parameter 'ap' here}}
void vmyprintf(const char *f, my_va_list ap);
void myprintf(const char *f, ...) {
  my_va_list ap;
  if (1) {
    __va_start(&ap, f);
    vmyprintf(f, ap);
    ap = 0;
  } else {
    __va_start(ap, f); // expected-warning {{incompatible pointer types passing 'my_va_list'}}
  }
}

// __unaligned handling
void test_unaligned() {
  __unaligned int *p1 = 0;
  int *p2 = p1; // expected-warning {{initializing 'int *' with an expression of type '__unaligned int *' discards qualifiers}}
  __unaligned int *p3 = p2;
}

void test_unaligned2(int x[__unaligned 4]) {}

