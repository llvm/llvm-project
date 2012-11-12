// RUN: %clang_cc1 %s -fsyntax-only -Wno-unused-value -Wmicrosoft -verify -fms-extensions


struct A
{
   int a[];  /* expected-warning {{flexible array member 'a' in otherwise empty struct is a Microsoft extension}} */
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

struct test {
  int c;
  struct nested2;   // expected-warning {{anonymous structs are a Microsoft extension}}
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
}


typedef struct {
  UNKNOWN u; // expected-error {{unknown type name 'UNKNOWN'}}
} AA;

typedef struct {
  AA; // expected-warning {{anonymous structs are a Microsoft extension}}
} BB;

__declspec(deprecated("This is deprecated")) enum DE1 { one, two } e1; // expected-note {{'e1' declared here}}
struct __declspec(deprecated) DS1 { int i; float f; }; // expected-note {{declared here}}

#define MY_TEXT		"This is also deprecated"
__declspec(deprecated(MY_TEXT)) void Dfunc1( void ) {} // expected-note {{'Dfunc1' declared here}}

struct __declspec(deprecated(123)) DS2 {};	// expected-error {{argument to deprecated attribute was not a string literal}}

void test( void ) {
	e1 = one;	// expected-warning {{'e1' is deprecated: This is deprecated}}
	struct DS1 s = { 0 };	// expected-warning {{'DS1' is deprecated}}
	Dfunc1();	// expected-warning {{'Dfunc1' is deprecated: This is also deprecated}}

	enum DE1 no;	// no warning because E1 is not deprecated
}
