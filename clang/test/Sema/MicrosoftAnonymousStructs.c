// RUN: %clang_cc1 -triple i686-windows %s -fsyntax-only -Wno-unused-value -Wno-pointer-to-int-cast -Wmicrosoft -verify -fms-anonymous-structs
// RUN: %clang_cc1 -triple powerpc-ibm-aix %s -fsyntax-only -Wno-unused-value -Wno-pointer-to-int-cast -Wmicrosoft -verify -fms-anonymous-structs

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

void foo(void)
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
