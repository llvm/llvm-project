// RUN: %clang_cc1 -triple i686-windows %s -fsyntax-only -Wno-unused-value \
// RUN:   -Wno-pointer-to-int-cast -Wmicrosoft -verify=ms-anonymous -fms-anonymous-structs
// RUN: %clang_cc1 -triple powerpc-ibm-aix %s -fsyntax-only -Wno-unused-value \
// RUN:   -Wno-pointer-to-int-cast -Wmicrosoft -verify=ms-anonymous -fms-anonymous-structs
// RUN: %clang_cc1 -triple i686-windows %s -fsyntax-only -Wno-unused-value \
// RUN:   -Wno-pointer-to-int-cast -Wmicrosoft -verify=ms-anonymous -fms-extensions
// RUN: %clang_cc1 -triple i686-windows %s -fsyntax-only -Wno-unused-value \
// RUN:   -Wno-pointer-to-int-cast -Wmicrosoft -verify=ms-anonymous -fms-compatibility
// RUN: %clang_cc1 -triple i686-windows %s -fsyntax-only -Wno-unused-value \
// RUN:   -Wno-pointer-to-int-cast -Wmicrosoft -verify=ms-anonymous-dis
// Test that explicit -fno-ms-anonymous-structs does not enable the feature.
// RUN: %clang_cc1 -triple i686-windows %s -fsyntax-only -Wno-unused-value \
// RUN:   -Wno-pointer-to-int-cast -Wmicrosoft -verify=ms-anonymous-dis \
// RUN:   -fno-ms-anonymous-structs
// Test that explicit -fno-ms-anonymous-structs overrides earlier -fms-anonymous-structs.
// RUN: %clang_cc1 -triple i686-windows %s -fsyntax-only -Wno-unused-value \
// RUN:   -Wno-pointer-to-int-cast -Wmicrosoft -verify=ms-anonymous-dis \
// RUN:   -fms-anonymous-structs -fno-ms-anonymous-structs
// Test that explicit -fms-anonymous-structs overrides earlier -fno-ms-anonymous-structs.
// RUN: %clang_cc1 -triple i686-windows %s -fsyntax-only -Wno-unused-value \
// RUN:   -Wno-pointer-to-int-cast -Wmicrosoft -verify=ms-anonymous \
// RUN:   -fno-ms-anonymous-structs -fms-anonymous-structs
// Test that explicit -fno-ms-anonymous-structs overrides earlier -fms-extensions.
// RUN: %clang_cc1 -triple i686-windows %s -fsyntax-only -Wno-unused-value \
// RUN:   -Wno-pointer-to-int-cast -Wmicrosoft -verify=ms-anonymous-dis \
// RUN:   -fms-extensions -fno-ms-anonymous-structs
// Test that explicit -fno-ms-anonymous-structs overrides earlier -fms-compatibility.
// RUN: %clang_cc1 -triple i686-windows %s -fsyntax-only -Wno-unused-value \
// RUN:   -Wno-pointer-to-int-cast -Wmicrosoft -verify=ms-anonymous-dis \
// RUN:   -fms-compatibility -fno-ms-anonymous-structs

struct union_mem {
  long g;
};

typedef struct nested1 {
  long a;
} NESTED1;

struct nested2 {
  long b;
  NESTED1;         // ms-anonymous-warning {{anonymous structs are a Microsoft extension}}
                   // ms-anonymous-dis-warning@-1 {{declaration does not declare anything}}
};

typedef union nested3 {
  long f;
  struct union_mem; // ms-anonymous-warning {{anonymous structs are a Microsoft extension}}
                    // ms-anonymous-dis-warning@-1 {{declaration does not declare anything}}
} NESTED3;

struct test {
  int c;
  struct nested2;   // ms-anonymous-warning {{anonymous structs are a Microsoft extension}}
                    // ms-anonymous-dis-warning@-1 {{declaration does not declare anything}}
  NESTED3;          // ms-anonymous-warning {{anonymous unions are a Microsoft extension}}
                    // ms-anonymous-dis-warning@-1 {{declaration does not declare anything}}
};

struct nested4 {
  long d;
  struct nested5 { // ms-anonymous-warning {{anonymous structs are a Microsoft extension}}
                   // ms-anonymous-dis-warning@-1 {{declaration does not declare anything}}
    long e;
  };
};

void foo(void)
{
  struct test var;
  var.c;
  var.a;          // ms-anonymous-dis-error {{no member named 'a' in 'struct test'}}
  var.b;          // ms-anonymous-dis-error {{no member named 'b' in 'struct test'}}
  var.f;          // ms-anonymous-dis-error {{no member named 'f' in 'struct test'}}
  var.g;          // ms-anonymous-dis-error {{no member named 'g' in 'struct test'}}
}
