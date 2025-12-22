// RUN: %clang_cc1 %s      -std=c++23 -triple=x86_64-apple-darwin10 -fsyntax-only -verify=expected,bitfieldwarnings,cpp -Wno-unused-value -Wno-unused-but-set-variable -Wbitfield-width -Wbitfield-enum-conversion
// RUN: %clang_cc1 %s      -std=c++23 -triple=x86_64-apple-darwin10 -fsyntax-only -verify=expected,cpp -Wno-unused-value -Wno-unused-but-set-variable

// RUN: %clang_cc1 %s -x c -std=c23   -triple=x86_64-apple-darwin10 -fsyntax-only -verify=expected,c -Wno-unused-value -Wno-unused-but-set-variable
// RUN: %clang_cc1 %s -x c -std=c23   -triple=x86_64-apple-darwin10 -fsyntax-only -verify=expected,bitfieldwarnings,c -Wno-unused-value -Wno-unused-but-set-variable -Wbitfield-width -Wbitfield-enum-conversion


typedef enum A {
  A_a,
  A_b,
  A_c,
  A_d
} A;

#ifdef __cplusplus
#define DEFINE_ENUM(_Name, _Type, ...) enum class _Name : _Type { __VA_ARGS__ } ;
#define ENUM_CLASS_REF(_Name, _Enum) _Name::_Enum
#else
#define DEFINE_ENUM(_Name, _Type, ...) typedef enum _Name : _Type { __VA_ARGS__ } _Name;
#define ENUM_CLASS_REF(_Name, _Enum) _Enum
#endif

DEFINE_ENUM(B, int, B_a, B_b, B_c, B_d );
DEFINE_ENUM(C, unsigned, C_a, C_b, C_c, C_d );
DEFINE_ENUM(D, unsigned, D_a, D_b);

// Not using templates here so we can more easily distinguish the responsible
// party for each warning

typedef struct S_A {
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
  D field10 : 1; // #S_A_field10
  __attribute__((preferred_type(A))) // #preferred_S_A_field11
  D field11 : 2; // #S_A_field11
  __attribute__((preferred_type(A)))
  D field12 : 8; // #S_A_field12
} S_A;

typedef struct S_B {
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
  D field10 : 1; // #S_B_field10
  __attribute__((preferred_type(B))) // #preferred_S_B_field11
  D field11 : 2; // #S_B_field11
  __attribute__((preferred_type(B)))
  D field12 : 8; // #S_B_field12
} S_B;

typedef struct S_C {
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
  D field10 : 1; // #S_C_field10
  __attribute__((preferred_type(C))) // #preferred_S_C_field11
  D field11 : 2; // #S_C_field11
  __attribute__((preferred_type(C)))
  D field12 : 8; // #S_C_field12
} S_C;

void read_enumA(S_A *s) {
  A x;
  x = s->field1;
  x = s->field2;
  x = s->field3;
  x = (A)s->field4;
  x = (A)s->field5;
  x = (A)s->field6;
  x = (A)s->field7;
  x = (A)s->field8;
  x = (A)s->field9;
  x = (A)s->field10;
  x = (A)s->field11;
  x = (A)s->field12;
}

void read_enumB(S_B *s) {
  B x;
  x = s->field1;
  x = s->field2;
  x = s->field3;
  x = (B)s->field4;
  x = (B)s->field5;
  x = (B)s->field6;
  x = (B)s->field7;
  x = (B)s->field8;
  x = (B)s->field9;
  x = (B)s->field10;
  x = (B)s->field11;
  x = (B)s->field12;
}

void read_enumC(S_C *s) {
  C x;
  x = s->field1;
  x = s->field2;
  x = s->field3;
  x = (C)s->field4;
  x = (C)s->field5;
  x = (C)s->field6;
  x = (C)s->field7;
  x = (C)s->field8;
  x = (C)s->field9;
  x = (C)s->field10;
  x = (C)s->field11;
  x = (C)s->field12;
}

void write_enumA(S_A *s, A x) {
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
  s->field10 = (D)x;
  s->field11 = (D)x;
  s->field12 = (D)x;
}

void write_enumB(S_B *s, B x) {
  s->field1 = x;
  // bitfieldwarnings-warning@-1 {{bit-field 'field1' is not wide enough to store all enumerators of 'B'}}
  // bitfieldwarnings-note@#S_B_field1 {{widen this field to 2 bits to store all values of 'B'}}
  s->field2 = x;
  // bitfieldwarnings-warning@-1 {{signed bit-field 'field2' needs an extra bit to represent the largest positive enumerators of 'B'}}
  // bitfieldwarnings-note@#S_B_field2 {{consider making the bit-field type unsigned}}
  s->field3 = x;
  s->field4 = (unsigned)x;
  // expected-warning@-1 {{bit-field 'field4' is not wide enough to store all enumerators of preferred type 'B'}}
  // expected-note@#S_B_field4 {{widen this field to 2 bits to store all values of 'B'}}
  // expected-note@#preferred_S_B_field4 {{preferred type for bit-field 'B' specified here}}
  s->field5 = (unsigned)x;
  s->field6 = (unsigned)x;
  s->field7 = (int)x;
  // expected-warning@-1 {{bit-field 'field7' is not wide enough to store all enumerators of preferred type 'B'}}
  // expected-note@#S_B_field7 {{widen this field to 2 bits to store all values of 'B'}}
  // expected-note@#preferred_S_B_field7 {{preferred type for bit-field 'B' specified here}}
  s->field8 = (int)x;
  // expected-warning@-1 {{signed bit-field 'field8' needs an extra bit to represent the largest positive enumerators of preferred type 'B'}}
  // expected-note@#S_B_field8 {{consider making the bit-field type unsigned}}
  // expected-note@#preferred_S_B_field8 {{preferred type for bit-field 'B' specified here}}
  s->field9 = (int)x;
  s->field10 = (D)x;
  s->field11 = (D)x;
  s->field12 = (D)x;
}

void write_enumC(S_C *s, C x) {
  s->field1 = x;
  // bitfieldwarnings-warning@-1 {{bit-field 'field1' is not wide enough to store all enumerators of 'C'}}
  // bitfieldwarnings-note@#S_C_field1 {{widen this field to 2 bits to store all values of 'C'}}
  s->field2 = x;
  s->field3 = x;
  s->field4 = (unsigned)x;
  // expected-warning@-1 {{bit-field 'field4' is not wide enough to store all enumerators of preferred type 'C'}}
  // expected-note@#S_C_field4 {{widen this field to 2 bits to store all values of 'C'}}
  // expected-note@#preferred_S_C_field4 {{preferred type for bit-field 'C' specified here}}
  s->field5 = (unsigned)x;
  s->field6 = (unsigned)x;
  s->field7 = (int)x;
  // expected-warning@-1 {{bit-field 'field7' is not wide enough to store all enumerators of preferred type 'C'}}
  // expected-note@#S_C_field7 {{widen this field to 2 bits to store all values of 'C'}}
  // expected-note@#preferred_S_C_field7 {{preferred type for bit-field 'C' specified here}}
  s->field8 = (int)x;
  // expected-warning@-1 {{signed bit-field 'field8' needs an extra bit to represent the largest positive enumerators of preferred type 'C'}}
  // expected-note@#S_C_field8 {{consider making the bit-field type unsigned}}
  // expected-note@#preferred_S_C_field8 {{preferred type for bit-field 'C' specified here}}
  s->field9 = (int)x;
  s->field10 = (D)x;
  s->field11 = (D)x;
  s->field12 = (D)x;
}

void write_enum_intA(struct S_A *s, int x) {
  s->field1 = (A)x;
  // bitfieldwarnings-warning@-1 {{bit-field 'field1' is not wide enough to store all enumerators of 'A'}}
  // bitfieldwarnings-note@#S_A_field1 {{widen this field to 2 bits to store all values of 'A'}}
  s->field2 = (A)x;
  s->field3 = (A)x;
  s->field4 = x;
  // expected-warning@-1 {{bit-field 'field4' is not wide enough to store all enumerators of preferred type 'A'}}
  // expected-note@#S_A_field4 {{widen this field to 2 bits to store all values of 'A'}}
  // expected-note@#preferred_S_A_field4 {{preferred type for bit-field 'A' specified here}}
  s->field5 = x;
  s->field6 = x;
  s->field7 = x;
  // expected-warning@-1 {{bit-field 'field7' is not wide enough to store all enumerators of preferred type 'A'}}
  // expected-note@#S_A_field7 {{widen this field to 2 bits to store all values of 'A'}}
  // expected-note@#preferred_S_A_field7 {{preferred type for bit-field 'A' specified here}}
  s->field8 = x;
  // expected-warning@-1 {{signed bit-field 'field8' needs an extra bit to represent the largest positive enumerators of preferred type 'A'}}
  // expected-note@#S_A_field8 {{consider making the bit-field type unsigned}}
  // expected-note@#preferred_S_A_field8 {{preferred type for bit-field 'A' specified here}}
  s->field9 = x;
  s->field10 = (D)x;
  s->field11 = (D)x;
  s->field12 = (D)x;
}

void write_enum_intB(struct S_B *s, int x) {
  s->field1 = (B)x;
  // bitfieldwarnings-warning@-1 {{bit-field 'field1' is not wide enough to store all enumerators of 'B'}}
  // bitfieldwarnings-note@#S_B_field1 {{widen this field to 2 bits to store all values of 'B'}}
  s->field2 = (B)x;
  // bitfieldwarnings-warning@-1 {{signed bit-field 'field2' needs an extra bit to represent the largest positive enumerators of 'B'}}
  // bitfieldwarnings-note@#S_B_field2 {{consider making the bit-field type unsigned}}
  s->field3 = (B)x;
  s->field4 = x;
  // expected-warning@-1 {{bit-field 'field4' is not wide enough to store all enumerators of preferred type 'B'}}
  // expected-note@#S_B_field4 {{widen this field to 2 bits to store all values of 'B'}}
  // expected-note@#preferred_S_B_field4 {{preferred type for bit-field 'B' specified here}}
  s->field5 = x;
  s->field6 = x;
  s->field7 = x;
  // expected-warning@-1 {{bit-field 'field7' is not wide enough to store all enumerators of preferred type 'B'}}
  // expected-note@#S_B_field7 {{widen this field to 2 bits to store all values of 'B'}}
  // expected-note@#preferred_S_B_field7 {{preferred type for bit-field 'B' specified here}}
  s->field8 = x;
  // expected-warning@-1 {{signed bit-field 'field8' needs an extra bit to represent the largest positive enumerators of preferred type 'B'}}
  // expected-note@#S_B_field8 {{consider making the bit-field type unsigned}}
  // expected-note@#preferred_S_B_field8 {{preferred type for bit-field 'B' specified here}}
  s->field9 = x;
  s->field10 = (D)x;
  s->field11 = (D)x;
  s->field12 = (D)x;
}

void write_enum_intC(struct S_C *s, int x) {
  s->field1 = (C)x;
  // bitfieldwarnings-warning@-1 {{bit-field 'field1' is not wide enough to store all enumerators of 'C'}}
  // bitfieldwarnings-note@#S_C_field1 {{widen this field to 2 bits to store all values of 'C'}}
  s->field2 = (C)x;
  s->field3 = (C)x;
  s->field4 = x;
  // expected-warning@-1 {{bit-field 'field4' is not wide enough to store all enumerators of preferred type 'C'}}
  // expected-note@#S_C_field4 {{widen this field to 2 bits to store all values of 'C'}}
  // expected-note@#preferred_S_C_field4 {{preferred type for bit-field 'C' specified here}}
  s->field5 = x;
  s->field6 = x;
  s->field7 = x;
  // expected-warning@-1 {{bit-field 'field7' is not wide enough to store all enumerators of preferred type 'C'}}
  // expected-note@#S_C_field7 {{widen this field to 2 bits to store all values of 'C'}}
  // expected-note@#preferred_S_C_field7 {{preferred type for bit-field 'C' specified here}}
  s->field8 = x;
  // expected-warning@-1 {{signed bit-field 'field8' needs an extra bit to represent the largest positive enumerators of preferred type 'C'}}
  // expected-note@#S_C_field8 {{consider making the bit-field type unsigned}}
  // expected-note@#preferred_S_C_field8 {{preferred type for bit-field 'C' specified here}}
  s->field9 = x;
  s->field10 = (D)x;
  s->field11 = (D)x;
  s->field12 = (D)x;
}

void write_low_constantA(S_A *s) {
  s->field1 = A_a;
  s->field2 = A_a;
  s->field3 = A_a;
  s->field4 = A_a;
  s->field5 = A_a;
  s->field6 = A_a;
  s->field7 = A_a;
  s->field8 = A_a;
  s->field9 = A_a;
  s->field10 = (D)A_a;
  s->field11 = (D)A_a;
  s->field12 = (D)A_a;
};

void write_low_constantB(S_B *s) {
  s->field1 = ENUM_CLASS_REF(B, B_a);
  s->field2 = ENUM_CLASS_REF(B, B_a);
  s->field3 = ENUM_CLASS_REF(B, B_a);
  s->field4 = (unsigned)ENUM_CLASS_REF(B, B_a);
  s->field5 = (unsigned)ENUM_CLASS_REF(B, B_a);
  s->field6 = (unsigned)ENUM_CLASS_REF(B, B_a);
  s->field7 = (int)ENUM_CLASS_REF(B, B_a);
  s->field8 = (int)ENUM_CLASS_REF(B, B_a);
  s->field9 = (int)ENUM_CLASS_REF(B, B_a);
  s->field10 = (D)ENUM_CLASS_REF(B, B_a);
  s->field11 = (D)ENUM_CLASS_REF(B, B_a);
  s->field12 = (D)ENUM_CLASS_REF(B, B_a);
};

void write_low_constantC(S_C *s) {
  s->field1 = ENUM_CLASS_REF(C, C_a);
  s->field2 = ENUM_CLASS_REF(C, C_a);
  s->field3 = ENUM_CLASS_REF(C, C_a);
  s->field4 = (unsigned)ENUM_CLASS_REF(C, C_a);
  s->field5 = (unsigned)ENUM_CLASS_REF(C, C_a);
  s->field6 = (unsigned)ENUM_CLASS_REF(C, C_a);
  s->field7 = (int)ENUM_CLASS_REF(C, C_a);
  s->field8 = (int)ENUM_CLASS_REF(C, C_a);
  s->field9 = (int)ENUM_CLASS_REF(C, C_a);
  s->field10 = (D)ENUM_CLASS_REF(C, C_a);
  s->field11 = (D)ENUM_CLASS_REF(C, C_a);
  s->field12 = (D)ENUM_CLASS_REF(C, C_a);
};

void write_high_constantA(S_A *s) {
  s->field1 = A_d;
  // cpp-warning@-1 {{implicit truncation from 'A' to bit-field changes value from 3 to 1}}
  // c-warning@-2 {{implicit truncation from 'int' to bit-field changes value from 3 to 1}}
  s->field2 = A_d;
  s->field3 = A_d;
  s->field4 = A_d;
  // cpp-warning@-1 {{implicit truncation from 'A' to bit-field changes value from 3 to 1}}
  // c-warning@-2 {{implicit truncation from 'int' to bit-field changes value from 3 to 1}}
  s->field5 = A_d;
  s->field6 = A_d;
  s->field7 = A_d;
  // cpp-warning@-1 {{implicit truncation from 'A' to bit-field changes value from 3 to -1}}
  // c-warning@-2 {{implicit truncation from 'int' to bit-field changes value from 3 to -1}}
  s->field8 = A_d;
  // cpp-warning@-1 {{implicit truncation from 'A' to bit-field changes value from 3 to -1}}
  // c-warning@-2 {{implicit truncation from 'int' to bit-field changes value from 3 to -1}}
  s->field9 = A_d;
  s->field10 = (D)A_d;
  // cpp-warning@-1 {{implicit truncation from 'D' to bit-field changes value from 3 to 1}}
  // c-warning@-2 {{implicit truncation from 'D' (aka 'enum D') to bit-field changes value from 3 to 1}}
  s->field11 = (D)A_d;
  s->field12 = (D)A_d;
};

void write_high_constantB(S_B *s) {
  s->field1 = ENUM_CLASS_REF(B, B_d);
  // cpp-warning@-1 {{implicit truncation from 'B' to bit-field changes value from 3 to 1}}
  // c-warning@-2 {{implicit truncation from 'int' to bit-field changes value from 3 to -1}}
  s->field2 = ENUM_CLASS_REF(B, B_d);
  // c-warning@-1 {{implicit truncation from 'int' to bit-field changes value from 3 to -1}}
  s->field3 = ENUM_CLASS_REF(B, B_d);
  s->field4 = (unsigned)ENUM_CLASS_REF(B, B_d);
  // expected-warning@-1 {{implicit truncation from 'unsigned int' to bit-field changes value from 3 to 1}}
  s->field5 = (unsigned)ENUM_CLASS_REF(B, B_d);
  s->field6 = (unsigned)ENUM_CLASS_REF(B, B_d);
  s->field7 = (int)ENUM_CLASS_REF(B, B_d);
  // expected-warning@-1 {{implicit truncation from 'int' to bit-field changes value from 3 to -1}}
  s->field8 = (int)ENUM_CLASS_REF(B, B_d);
  // expected-warning@-1 {{implicit truncation from 'int' to bit-field changes value from 3 to -1}}
  s->field9 = (int)ENUM_CLASS_REF(B, B_d);
};


void write_high_constantC(S_C *s) {
  s->field1 = ENUM_CLASS_REF(C, C_d);
  // cpp-warning@-1 {{implicit truncation from 'C' to bit-field changes value from 3 to 1}}
  // c-warning@-2 {{implicit truncation from 'unsigned int' to bit-field changes value from 3 to 1}}
  s->field2 = ENUM_CLASS_REF(C, C_d);
  s->field3 = ENUM_CLASS_REF(C, C_d);
  s->field4 = (unsigned)ENUM_CLASS_REF(C, C_d);
  // expected-warning@-1 {{implicit truncation from 'unsigned int' to bit-field changes value from 3 to 1}}
  s->field5 = (unsigned)ENUM_CLASS_REF(C, C_d);
  s->field6 = (unsigned)ENUM_CLASS_REF(C, C_d);
  s->field7 = (int)ENUM_CLASS_REF(C, C_d);
  // expected-warning@-1 {{implicit truncation from 'int' to bit-field changes value from 3 to -1}}
  s->field8 = (int)ENUM_CLASS_REF(C, C_d);
  // expected-warning@-1 {{implicit truncation from 'int' to bit-field changes value from 3 to -1}}
  s->field9 = (int)ENUM_CLASS_REF(C, C_d);
};
