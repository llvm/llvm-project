// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -fsyntax-only -verify=expected,preferrednotes -std=c11 -Wno-unused-value -Wno-unused-but-set-variable
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -fsyntax-only -verify=expected,bitfieldwarning,preferrednotes -std=c11 -Wno-unused-value -Wno-unused-but-set-variable -Wbitfield-width -Wbitfield-enum-conversion
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -fsyntax-only -verify=noerror,preferrednotes -std=c11 -Wno-unused-value -Wno-unused-but-set-variable -Wno-error=preferred-type-bitfield-enum-conversion -Wno-error=preferred-type-bitfield-width

enum A {
  A_a,
  A_b,
  A_c,
  A_d
};

struct S {
  enum A a1 : 1; // #S_a1_decl
  enum A a2 : 2;
  enum A a3 : 8;
  __attribute__((preferred_type(enum A))) // #preferred_S_a4
  unsigned a4 : 1; // #S_a4_decl
  __attribute__((preferred_type(enum A)))
  unsigned a5 : 2;
  __attribute__((preferred_type(enum A)))
  unsigned a6 : 8;
  __attribute__((preferred_type(enum A)))  // #preferred_S_a7
  int a7 : 1; // #S_a7_decl
  __attribute__((preferred_type(enum A))) // #preferred_S_a8
  int a8 : 2; // #S_a8_decl
  __attribute__((preferred_type(enum A)))
  int a9 : 8;
};

void read_enum(struct S *s) {
  enum A x;
  x = s->a1;
  x = s->a2;
  x = s->a3;
  x = s->a4;
  x = s->a5;
  x = s->a6;
  x = s->a7;
  x = s->a8;
  x = s->a9;
}

void write_enum(struct S *s, enum A x) {
  s->a1 = x;
  // bitfieldwarning-warning@-1 {{bit-field 'a1' is not wide enough to store all enumerators of 'A'}}
  // bitfieldwarning-note@#S_a1_decl {{widen this field to 2 bits to store all values of 'A'}}
  s->a2 = x;
  s->a3 = x;
  s->a4 = x;
  // bitfieldwarning-warning@-1 {{bit-field 'a4' is not wide enough to store all enumerators of 'A'}}
  // bitfieldwarning-note@#S_a4_decl {{widen this field to 2 bits to store all values of 'A'}}
  s->a5 = x;
  s->a6 = x;
  s->a7 = x;
  // bitfieldwarning-warning@-1 {{bit-field 'a7' is not wide enough to store all enumerators of 'A'}}
  // bitfieldwarning-note@#S_a7_decl {{widen this field to 2 bits to store all values of 'A'}}
  s->a8 = x;
  // bitfieldwarning-warning@-1 {{signed bit-field 'a8' needs an extra bit to represent the largest positive enumerators of 'A'}}
  // bitfieldwarning-note@#S_a8_decl {{consider making the bit-field type unsigned}}
  s->a9 = x;
}

void write_enum_int(struct S *s, int x) {
  s->a1 = x;
  s->a2 = x;
  s->a3 = x;
  s->a4 = x;
  // expected-error@-1 {{bit-field 'a4' is not wide enough to store all enumerators of preferred type 'A'}}
  // noerror-warning@-2 {{bit-field 'a4' is not wide enough to store all enumerators of preferred type 'A'}}
  // preferrednotes-note@#S_a4_decl {{widen this field to 2 bits to store all values of 'A'}}
  // preferrednotes-note@#preferred_S_a4 {{preferred type for bit-field 'A' specified here}}
  s->a5 = x;
  s->a6 = x;
  s->a7 = x;
  // expected-error@-1 {{bit-field 'a7' is not wide enough to store all enumerators of preferred type 'A'}}
  // noerror-warning@-2 {{bit-field 'a7' is not wide enough to store all enumerators of preferred type 'A'}}
  // preferrednotes-note@#S_a7_decl {{widen this field to 2 bits to store all values of 'A'}}
  // preferrednotes-note@#preferred_S_a7 {{preferred type for bit-field 'A' specified here}}
  s->a8 = x;
  // expected-error@-1 {{signed bit-field 'a8' needs an extra bit to represent the largest positive enumerators of preferred type 'A'}}
  // noerror-warning@-2 {{signed bit-field 'a8' needs an extra bit to represent the largest positive enumerators of preferred type 'A'}}
  // preferrednotes-note@#S_a8_decl {{consider making the bit-field type unsigned}}
  // preferrednotes-note@#preferred_S_a8 {{preferred type for bit-field 'A' specified here}}
  s->a9 = x;
}

void write_low_constant(struct S *s) {
  s->a1 = A_a;
  s->a2 = A_a;
  s->a3 = A_a;
  s->a4 = A_a;
  s->a5 = A_a;
  s->a6 = A_a;
  s->a7 = A_a;
  s->a8 = A_a;
  s->a9 = A_a;
};

void write_high_constant(struct S *s) {
  s->a1 = A_d;
  // preferrednotes-warning@-1 {{implicit truncation from 'int' to bit-field changes value from 3 to 1}}
  s->a2 = A_d;
  s->a3 = A_d;
  s->a4 = A_d;
  // preferrednotes-warning@-1 {{implicit truncation from 'int' to bit-field changes value from 3 to 1}}
  s->a5 = A_d;
  s->a6 = A_d;
  s->a7 = A_d;
  // preferrednotes-warning@-1 {{implicit truncation from 'int' to bit-field changes value from 3 to -1}}
  s->a8 = A_d;
  // preferrednotes-warning@-1 {{implicit truncation from 'int' to bit-field changes value from 3 to -1}}
  s->a9 = A_d;
};
