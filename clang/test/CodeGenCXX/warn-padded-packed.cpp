// RUN: %clang_cc1 -triple=x86_64-none-none -Wpadded -Wpacked -verify %s -emit-llvm-only

struct S1 {
  char c;
  short s; // expected-warning {{padding struct 'S1' with 1 byte to align 's'}}
  long l; // expected-warning {{padding struct 'S1' with 4 bytes to align 'l'}}
};

struct S2 { // expected-warning {{padding size of 'S2' with 3 bytes to alignment boundary}}
  int i;
  char c;
};

struct S3 {
  char c;
  int i;
} __attribute__((packed));

struct S4 {
  int i; // expected-warning {{packed attribute is unnecessary for 'i'}}
  char c;
} __attribute__((packed));

struct S5 {
  char c;
  union {
    char c;
    int i;
  } u; // expected-warning {{padding struct 'S5' with 3 bytes to align 'u'}}
};

struct S6 { // expected-warning {{padding size of 'S6' with 30 bits to alignment boundary}}
  int i : 2;
};

struct S7 { // expected-warning {{padding size of 'S7' with 7 bytes to alignment boundary}}
  char c;
  virtual void m();
};

struct B {
  char c;
};

struct S8 : B {
  int i; // expected-warning {{padding struct 'S8' with 3 bytes to align 'i'}}
};

struct S9 { // expected-warning {{packed attribute is unnecessary for 'S9'}}
  int x; // expected-warning {{packed attribute is unnecessary for 'x'}}
  int y; // expected-warning {{packed attribute is unnecessary for 'y'}}
} __attribute__((packed));

struct S10 { // expected-warning {{packed attribute is unnecessary for 'S10'}}
  int x; // expected-warning {{packed attribute is unnecessary for 'x'}}
  char a,b,c,d;
} __attribute__((packed));


struct S11 {
  bool x;
  char a,b,c,d;
} __attribute__((packed));

struct S12 {
  bool b : 1;
  char c; // expected-warning {{padding struct 'S12' with 7 bits to align 'c'}}
};

struct S13 { // expected-warning {{padding size of 'S13' with 6 bits to alignment boundary}}
  char c;
  bool b : 10; // expected-warning {{size of bit-field 'b' (10 bits) exceeds the size of its type}}
};

// The warnings are emitted when the layout of the structs is computed, so we have to use them.
void f(S1*, S2*, S3*, S4*, S5*, S6*, S7*, S8*, S9*, S10*, S11*, S12*, S13*) { }
