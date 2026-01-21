// RUN: %clang_cc1 -triple i686-windows %s -fsyntax-only -Wno-unused-value \
// RUN: -Wno-pointer-to-int-cast -Wmicrosoft -verify=ms-anonymous -fms-anonymous-structs
// RUN: %clang_cc1 -triple powerpc-ibm-aix %s -fsyntax-only -Wno-unused-value \
// RUN: -Wno-pointer-to-int-cast -Wmicrosoft -verify=ms-anonymous -fms-anonymous-structs
// RUN: %clang_cc1 -triple i686-windows %s -fsyntax-only -Wno-unused-value \
// RUN: -Wno-pointer-to-int-cast -Wmicrosoft -verify=ms-anonymous -fms-extensions
// RUN: %clang_cc1 -triple i686-windows %s -fsyntax-only -Wno-unused-value \
// RUN: -Wno-pointer-to-int-cast -Wmicrosoft -verify=ms-anonymous -fms-compatibility
// RUN: %clang_cc1 -triple i686-windows %s -fsyntax-only -Wno-unused-value \
// RUN: -Wno-pointer-to-int-cast -Wmicrosoft -verify=ms-anonymous-dis

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
  NESTED1;  // ms-anonymous-warning {{anonymous structs are a Microsoft extension}}
            // ms-anonymous-dis-warning@-1 {{declaration does not declare anything}}
};

struct nested2 PR20573 = { .a = 3 };  // ms-anonymous-dis-error {{field designator 'a' does not refer to any field in type 'struct nested2'}}

struct nested3 {
  long d;
  struct nested4 { // ms-anonymous-warning {{anonymous structs are a Microsoft extension}}
                   // ms-anonymous-dis-warning@-1 {{declaration does not declare anything}}
    long e;
  };
  union nested5 { // ms-anonymous-warning {{anonymous unions are a Microsoft extension}}
                  // ms-anonymous-dis-warning@-1 {{declaration does not declare anything}}
    long f;
  };
};

typedef union nested6 {
  long f;
} NESTED6;

struct test {
  int c;
  struct nested2;   // ms-anonymous-warning {{anonymous structs are a Microsoft extension}}
                    // ms-anonymous-dis-warning@-1 {{declaration does not declare anything}}
  NESTED6;   // ms-anonymous-warning {{anonymous unions are a Microsoft extension}}
            // ms-anonymous-dis-warning@-1 {{declaration does not declare anything}}
};

void foo(void)
{
  struct test var;
  var.a;      // ms-anonymous-dis-error {{no member named 'a' in 'struct test'}}
  var.b;      // ms-anonymous-dis-error {{no member named 'b' in 'struct test'}}
  var.c;
  var.bad1;   // ms-anonymous-error {{no member named 'bad1' in 'struct test'}}
              // ms-anonymous-dis-error@-1 {{no member named 'bad1' in 'struct test'}}
  var.bad2;   // ms-anonymous-error {{no member named 'bad2' in 'struct test'}}
              // ms-anonymous-dis-error@-1 {{no member named 'bad2' in 'struct test'}}
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
  UNKNOWN u; // ms-anonymous-error {{unknown type name 'UNKNOWN'}}
             // ms-anonymous-dis-error@-1 {{unknown type name 'UNKNOWN'}}
} AA;

typedef struct {
  AA; // ms-anonymous-warning {{anonymous structs are a Microsoft extension}}
      // ms-anonymous-dis-warning@-1 {{declaration does not declare anything}}
} BB;

struct anon_fault {
  struct undefined; // ms-anonymous-warning {{anonymous structs are a Microsoft extension}}
                    // ms-anonymous-error@-1 {{field has incomplete type 'struct undefined'}}
                    // ms-anonymous-note@-2 {{forward declaration of 'struct undefined'}}
                    // ms-anonymous-dis-warning@-3 {{declaration does not declare anything}}
};

const int anon_falt_size = sizeof(struct anon_fault);
