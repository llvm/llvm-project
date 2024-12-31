// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -fsyntax-only -verify -std=c++23 -Wno-unused-value -Wno-unused-but-set-variable
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -fsyntax-only -verify=expected,bitfieldwarnings -std=c++23 -Wno-unused-value -Wno-unused-but-set-variable -Wbitfield-width -Wbitfield-enum-conversion

// This is more complex than the C version because the user can specify the
// storage type 

enum A {
  A_a,
  A_b,
  A_c,
  A_d
};

enum class B : int {
  a,
  b,
  c,
  d
};

enum class C : unsigned {
  a,
  b,
  c,
  d
};

enum class Derp : unsigned {
  a,
  b
};

// Not using templates here so we can more easily distinguish the responsible
// party for each warning

struct S_A {
  A field1 : 1; // #S_A_field1
  A field2 : 2; // #S_A_field2
  A field3 : 8; // #S_A_field3
  __attribute__((preferred_type(A))) // #preferred_S_A_field4
  unsigned field4 : 1; // #S_A_field4
  __attribute__((preferred_type(A)))
  unsigned field5 : 2; // #S_A_field5
  __attribute__((preferred_type(A)))
  unsigned field6 : 8; // #S_A_field6
  __attribute__((preferred_type(A))) // #preferred_S_A_field7
  int field7 : 1; // #S_A_field7
  __attribute__((preferred_type(A))) // #preferred_S_A_field8
  int field8 : 2; // #S_A_field8
  __attribute__((preferred_type(A)))
  int field9 : 8; // #S_A_field9
  __attribute__((preferred_type(A)))
  Derp field10 : 1; // #S_A_field10
  __attribute__((preferred_type(A))) // #preferred_S_A_field11
  Derp field11 : 2; // #S_A_field11
  __attribute__((preferred_type(A)))
  Derp field12 : 8; // #S_A_field12
};

struct S_B {
  B field1 : 1; // #S_B_field1
  B field2 : 2; // #S_B_field2
  B field3 : 8; // #S_B_field3
  __attribute__((preferred_type(B))) // #preferred_S_B_field4
  unsigned field4 : 1; // #S_B_field4
  __attribute__((preferred_type(B)))
  unsigned field5 : 2; // #S_B_field5
  __attribute__((preferred_type(B)))
  unsigned field6 : 8; // #S_B_field6
  __attribute__((preferred_type(B))) // #preferred_S_B_field7
  int field7 : 1; // #S_B_field7
  __attribute__((preferred_type(B))) // #preferred_S_B_field8
  int field8 : 2; // #S_B_field8
  __attribute__((preferred_type(B)))
  int field9 : 8; // #S_B_field9
  __attribute__((preferred_type(B)))
  Derp field10 : 1; // #S_B_field10
  __attribute__((preferred_type(B))) // #preferred_S_B_field11
  Derp field11 : 2; // #S_B_field11
  __attribute__((preferred_type(B)))
  Derp field12 : 8; // #S_B_field12
};

struct S_C {
  C field1 : 1; // #S_C_field1
  C field2 : 2; // #S_C_field2
  C field3 : 8; // #S_C_field3
  __attribute__((preferred_type(C))) // #preferred_S_C_field4
  unsigned field4 : 1; // #S_C_field4
  __attribute__((preferred_type(C)))
  unsigned field5 : 2; // #S_C_field5
  __attribute__((preferred_type(C)))
  unsigned field6 : 8; // #S_C_field6
  __attribute__((preferred_type(C))) // #preferred_S_C_field7
  int field7 : 1; // #S_C_field7
  __attribute__((preferred_type(C))) // #preferred_S_C_field8
  int field8 : 2; // #S_C_field8
  __attribute__((preferred_type(C)))
  int field9 : 8; // #S_C_field9
  __attribute__((preferred_type(C)))
  Derp field10 : 1; // #S_C_field10
  __attribute__((preferred_type(C))) // #preferred_S_C_field11
  Derp field11 : 2; // #S_C_field11
  __attribute__((preferred_type(C)))
  Derp field12 : 8; // #S_C_field12
};

void read_enum(S_A *s) {
  using EnumType = A;
  EnumType x;
  x = s->field1;
  x = s->field2;
  x = s->field3;
  x = (EnumType)s->field4;
  x = (EnumType)s->field5;
  x = (EnumType)s->field6;
  x = (EnumType)s->field7;
  x = (EnumType)s->field8;
  x = (EnumType)s->field9;
  x = (EnumType)s->field10;
  x = (EnumType)s->field11;
  x = (EnumType)s->field12;
}

void read_enum(S_B *s) {
  using EnumType = B;
  EnumType x;
  x = s->field1;
  x = s->field2;
  x = s->field3;
  x = (EnumType)s->field4;
  x = (EnumType)s->field5;
  x = (EnumType)s->field6;
  x = (EnumType)s->field7;
  x = (EnumType)s->field8;
  x = (EnumType)s->field9;
  x = (EnumType)s->field10;
  x = (EnumType)s->field11;
  x = (EnumType)s->field12;
}

void read_enum(S_C *s) {
  using EnumType = C;
  EnumType x;
  x = s->field1;
  x = s->field2;
  x = s->field3;
  x = (EnumType)s->field4;
  x = (EnumType)s->field5;
  x = (EnumType)s->field6;
  x = (EnumType)s->field7;
  x = (EnumType)s->field8;
  x = (EnumType)s->field9;
  x = (EnumType)s->field10;
  x = (EnumType)s->field11;
  x = (EnumType)s->field12;
}

void write_enum(S_A *s, A x) {
  s->field1 = x;
  // bitfieldwarnings-warning@-1 {{bit-field 'field1' is not wide enough to store all enumerators of 'A'}}
  // bitfieldwarnings-note@#S_A_field1 {{widen this field to 2 bits to store all values of 'A'}}
  s->field2 = x;
  s->field3 = x;
  s->field4 = x;
  // bitfieldwarnings-warning@-1 {{bit-field 'field4' is not wide enough to store all enumerators of 'A'}}
  // bitfieldwarnings-note@#S_A_field4 {{widen this field to 2 bits to store all values of 'A'}}
  s->field5 = x;
  s->field6 = x;
  s->field7 = x;
  // bitfieldwarnings-warning@-1 {{bit-field 'field7' is not wide enough to store all enumerators of 'A'}}
  // bitfieldwarnings-note@#S_A_field7 {{widen this field to 2 bits to store all values of 'A'}}
  s->field8 = x;
  // bitfieldwarnings-warning@-1 {{signed bit-field 'field8' needs an extra bit to represent the largest positive enumerators of 'A'}}
  // bitfieldwarnings-note@#S_A_field8 {{consider making the bit-field type unsigned}}
  s->field9 = x;
  s->field10 = (Derp)x;
  s->field11 = (Derp)x;
  s->field12 = (Derp)x;
}

void write_enum(S_B *s, B x) {
  s->field1 = x;
  // bitfieldwarnings-warning@-1 {{bit-field 'field1' is not wide enough to store all enumerators of 'B'}}
  // bitfieldwarnings-note@#S_B_field1 {{widen this field to 2 bits to store all values of 'B'}}
  s->field2 = x;
  s->field3 = x;
  s->field4 = (unsigned)x;
  // expected-error@-1 {{bit-field 'field4' is not wide enough to store all enumerators of preferred type 'B'}}
  // expected-note@#S_B_field4 {{widen this field to 2 bits to store all values of 'B'}}
  // expected-note@#preferred_S_B_field4 {{preferred type for bit-field 'B' specified here}}
  s->field5 = (unsigned)x;
  s->field6 = (unsigned)x;
  s->field7 = (int)x;
  // expected-error@-1 {{bit-field 'field7' is not wide enough to store all enumerators of preferred type 'B'}}
  // expected-note@#S_B_field7 {{widen this field to 2 bits to store all values of 'B'}}
  // expected-note@#preferred_S_B_field7 {{preferred type for bit-field 'B' specified here}}
  s->field8 = (int)x;
  // expected-error@-1 {{signed bit-field 'field8' needs an extra bit to represent the largest positive enumerators of preferred type 'B'}}
  // expected-note@#S_B_field8 {{consider making the bit-field type unsigned}}
  // expected-note@#preferred_S_B_field8 {{preferred type for bit-field 'B' specified here}}
  s->field9 = (int)x;
  s->field10 = (Derp)x;
  s->field11 = (Derp)x;
  s->field12 = (Derp)x;
}
void write_enum(S_C *s, C x) {
  s->field1 = x;
  // bitfieldwarnings-warning@-1 {{bit-field 'field1' is not wide enough to store all enumerators of 'C'}}
  // bitfieldwarnings-note@#S_C_field1 {{widen this field to 2 bits to store all values of 'C'}}
  s->field2 = x;
  s->field3 = x;
  s->field4 = (unsigned)x;
  // expected-error@-1 {{bit-field 'field4' is not wide enough to store all enumerators of preferred type 'C'}}
  // expected-note@#S_C_field4 {{widen this field to 2 bits to store all values of 'C'}}
  // expected-note@#preferred_S_C_field4 {{preferred type for bit-field 'C' specified here}}
  s->field5 = (unsigned)x;
  s->field6 = (unsigned)x;
  s->field7 = (int)x;
  // expected-error@-1 {{bit-field 'field7' is not wide enough to store all enumerators of preferred type 'C'}}
  // expected-note@#S_C_field7 {{widen this field to 2 bits to store all values of 'C'}}
  // expected-note@#preferred_S_C_field7 {{preferred type for bit-field 'C' specified here}}
  s->field8 = (int)x;
  // expected-error@-1 {{signed bit-field 'field8' needs an extra bit to represent the largest positive enumerators of preferred type 'C'}}
  // expected-note@#S_C_field8 {{consider making the bit-field type unsigned}}
  // expected-note@#preferred_S_C_field8 {{preferred type for bit-field 'C' specified here}}
  s->field9 = (int)x;
  s->field10 = (Derp)x;
  s->field11 = (Derp)x;
  s->field12 = (Derp)x;
}

void write_enum_int(struct S_A *s, int x) {
  using EnumType = A;
  s->field1 = (EnumType)x;
  // bitfieldwarnings-warning@-1 {{bit-field 'field1' is not wide enough to store all enumerators of 'A'}}
  // bitfieldwarnings-note@#S_A_field1 {{widen this field to 2 bits to store all values of 'A'}}
  s->field2 = (EnumType)x;
  s->field3 = (EnumType)x;
  s->field4 = x;
  // expected-error@-1 {{bit-field 'field4' is not wide enough to store all enumerators of preferred type 'A'}}
  // expected-note@#S_A_field4 {{widen this field to 2 bits to store all values of 'A'}}
  // expected-note@#preferred_S_A_field4 {{preferred type for bit-field 'A' specified here}}
  s->field5 = x;
  s->field6 = x;
  s->field7 = x;
  // expected-error@-1 {{bit-field 'field7' is not wide enough to store all enumerators of preferred type 'A'}}
  // expected-note@#S_A_field7 {{widen this field to 2 bits to store all values of 'A'}}
  // expected-note@#preferred_S_A_field7 {{preferred type for bit-field 'A' specified here}}
  s->field8 = x;
  // expected-error@-1 {{signed bit-field 'field8' needs an extra bit to represent the largest positive enumerators of preferred type 'A'}}
  // expected-note@#S_A_field8 {{consider making the bit-field type unsigned}}
  // expected-note@#preferred_S_A_field8 {{preferred type for bit-field 'A' specified here}}
  s->field9 = x;
  s->field10 = (Derp)x;
  s->field11 = (Derp)x;
  s->field12 = (Derp)x;
}

void write_enum_int(struct S_B *s, int x) {
  using EnumType = B;
  s->field1 = (EnumType)x;
  // bitfieldwarnings-warning@-1 {{bit-field 'field1' is not wide enough to store all enumerators of 'B'}}
  // bitfieldwarnings-note@#S_B_field1 {{widen this field to 2 bits to store all values of 'B'}}
  s->field2 = (EnumType)x;
  s->field3 = (EnumType)x;
  s->field4 = x;
  // expected-error@-1 {{bit-field 'field4' is not wide enough to store all enumerators of preferred type 'B'}}
  // expected-note@#S_B_field4 {{widen this field to 2 bits to store all values of 'B'}}
  // expected-note@#preferred_S_B_field4 {{preferred type for bit-field 'B' specified here}}
  s->field5 = x;
  s->field6 = x;
  s->field7 = x;
  // expected-error@-1 {{bit-field 'field7' is not wide enough to store all enumerators of preferred type 'B'}}
  // expected-note@#S_B_field7 {{widen this field to 2 bits to store all values of 'B'}}
  // expected-note@#preferred_S_B_field7 {{preferred type for bit-field 'B' specified here}}
  s->field8 = x;
  // expected-error@-1 {{signed bit-field 'field8' needs an extra bit to represent the largest positive enumerators of preferred type 'B'}}
  // expected-note@#S_B_field8 {{consider making the bit-field type unsigned}}
  // expected-note@#preferred_S_B_field8 {{preferred type for bit-field 'B' specified here}}
  s->field9 = x;
  s->field10 = (Derp)x;
  s->field11 = (Derp)x;
  s->field12 = (Derp)x;
}

void write_enum_int(struct S_C *s, int x) {
  using EnumType = C;
  s->field1 = (EnumType)x;
  // bitfieldwarnings-warning@-1 {{bit-field 'field1' is not wide enough to store all enumerators of 'C'}}
  // bitfieldwarnings-note@#S_C_field1 {{widen this field to 2 bits to store all values of 'C'}}
  s->field2 = (EnumType)x;
  s->field3 = (EnumType)x;
  s->field4 = x;
  // expected-error@-1 {{bit-field 'field4' is not wide enough to store all enumerators of preferred type 'C'}}
  // expected-note@#S_C_field4 {{widen this field to 2 bits to store all values of 'C'}}
  // expected-note@#preferred_S_C_field4 {{preferred type for bit-field 'C' specified here}}
  s->field5 = x;
  s->field6 = x;
  s->field7 = x;
  // expected-error@-1 {{bit-field 'field7' is not wide enough to store all enumerators of preferred type 'C'}}
  // expected-note@#S_C_field7 {{widen this field to 2 bits to store all values of 'C'}}
  // expected-note@#preferred_S_C_field7 {{preferred type for bit-field 'C' specified here}}
  s->field8 = x;
  // expected-error@-1 {{signed bit-field 'field8' needs an extra bit to represent the largest positive enumerators of preferred type 'C'}}
  // expected-note@#S_C_field8 {{consider making the bit-field type unsigned}}
  // expected-note@#preferred_S_C_field8 {{preferred type for bit-field 'C' specified here}}
  s->field9 = x;
  s->field10 = (Derp)x;
  s->field11 = (Derp)x;
  s->field12 = (Derp)x;
}

void write_low_constant(S_A *s) {
  s->field1 = A_a;
  s->field2 = A_a;
  s->field3 = A_a;
  s->field4 = A_a;
  s->field5 = A_a;
  s->field6 = A_a;
  s->field7 = A_a;
  s->field8 = A_a;
  s->field9 = A_a;
  s->field10 = (Derp)A_a;
  s->field11 = (Derp)A_a;
  s->field12 = (Derp)A_a;
};

void write_low_constant(S_B *s) {
  using EnumType = B;
  s->field1 = EnumType::a;
  s->field2 = EnumType::a;
  s->field3 = EnumType::a;
  s->field4 = (unsigned)EnumType::a;
  s->field5 = (unsigned)EnumType::a;
  s->field6 = (unsigned)EnumType::a;
  s->field7 = (int)EnumType::a;
  s->field8 = (int)EnumType::a;
  s->field9 = (int)EnumType::a;
  s->field10 = (Derp)EnumType::a;
  s->field11 = (Derp)EnumType::a;
  s->field12 = (Derp)EnumType::a;
};

void write_low_constant(S_C *s) {
  using EnumType = C;
  s->field1 = EnumType::a;
  s->field2 = EnumType::a;
  s->field3 = EnumType::a;
  s->field4 = (unsigned)EnumType::a;
  s->field5 = (unsigned)EnumType::a;
  s->field6 = (unsigned)EnumType::a;
  s->field7 = (int)EnumType::a;
  s->field8 = (int)EnumType::a;
  s->field9 = (int)EnumType::a;
  s->field10 = (Derp)EnumType::a;
  s->field11 = (Derp)EnumType::a;
  s->field12 = (Derp)EnumType::a;
};

void write_high_constant(S_A *s) {
  s->field1 = A_d;
  // expected-warning@-1 {{implicit truncation from 'A' to bit-field changes value from 3 to 1}}
  s->field2 = A_d;
  s->field3 = A_d;
  s->field4 = A_d;
  // expected-warning@-1 {{implicit truncation from 'A' to bit-field changes value from 3 to 1}}
  s->field5 = A_d;
  s->field6 = A_d;
  s->field7 = A_d;
  // expected-warning@-1 {{implicit truncation from 'A' to bit-field changes value from 3 to -1}}
  s->field8 = A_d;
  // expected-warning@-1 {{implicit truncation from 'A' to bit-field changes value from 3 to -1}}
  s->field9 = A_d;
  s->field10 = (Derp)A_d;
  // expected-warning@-1 {{implicit truncation from 'Derp' to bit-field changes value from 3 to 1}}
  s->field11 = (Derp)A_d;
  s->field12 = (Derp)A_d;
};

void write_high_constant(S_B *s) {
  using EnumType = B;
  s->field1 = EnumType::d;
  // expected-warning@-1 {{implicit truncation from 'B' to bit-field changes value from 3 to 1}}
  s->field2 = EnumType::d;
  s->field3 = EnumType::d;
  s->field4 = (unsigned)EnumType::d;
  // expected-warning@-1 {{implicit truncation from 'unsigned int' to bit-field changes value from 3 to 1}}
  s->field5 = (unsigned)EnumType::d;
  s->field6 = (unsigned)EnumType::d;
  s->field7 = (int)EnumType::d;
  // expected-warning@-1 {{implicit truncation from 'int' to bit-field changes value from 3 to -1}}
  s->field8 = (int)EnumType::d;
  // expected-warning@-1 {{implicit truncation from 'int' to bit-field changes value from 3 to -1}}
  s->field9 = (int)EnumType::d;
};


void write_high_constant(S_C *s) {
  using EnumType = C;
  s->field1 = EnumType::d;
  // expected-warning@-1 {{implicit truncation from 'C' to bit-field changes value from 3 to 1}}
  s->field2 = EnumType::d;
  s->field3 = EnumType::d;
  s->field4 = (unsigned)EnumType::d;
  // expected-warning@-1 {{implicit truncation from 'unsigned int' to bit-field changes value from 3 to 1}}
  s->field5 = (unsigned)EnumType::d;
  s->field6 = (unsigned)EnumType::d;
  s->field7 = (int)EnumType::d;
  // expected-warning@-1 {{implicit truncation from 'int' to bit-field changes value from 3 to -1}}
  s->field8 = (int)EnumType::d;
  // expected-warning@-1 {{implicit truncation from 'int' to bit-field changes value from 3 to -1}}
  s->field9 = (int)EnumType::d;
};
