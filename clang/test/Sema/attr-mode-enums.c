// RUN: %clang_cc1 -fsyntax-only -verify %s

// Test checks that 'mode' attribute is handled correctly with enums, i. e. code
//   1. "typedef enum { A } __attribute__((mode(HI))) T;" is accepted,
//   2. "enum X __attribute__((mode(QI))) var;" forms a complete integer type.
//   3. "enum { A } __attribute__((mode(V4SI))) var;" is not accepted (vector mode).

typedef enum { E4 } EnumType;

int main() {
  // Vector mode are not allowed with enums.
  typedef enum { E1 } __attribute__((mode(V4QI))) RejectedType1; // expected-error{{mode 'V4QI' is not supported for enumeration types}}
  // expected-warning@-1{{specifying vector types with the 'mode' attribute is deprecated}}
  typedef enum __attribute__((mode(V8HI))) { E2 } RejectedType2; // expected-error{{mode 'V8HI' is not supported for enumeration types}}
                                                                 // expected-warning@-1{{deprecated}}
  typedef enum E3 __attribute__((mode(V2SI))) RejectedType3; // expected-error{{mode 'V2SI' is not supported for enumeration types}}
                                                             // expected-warning@-1{{deprecated}}
  typedef EnumType __attribute__((mode(V4DI))) RejectedType4; // expected-error{{mode 'V4DI' is not supported for enumeration types}}
                                                              // expected-warning@-1{{deprecated}}
  EnumType v1 __attribute__((mode(V4QI))); // expected-error{{mode 'V4QI' is not supported for enumeration types}}
                                           // expected-warning@-1{{deprecated}}
  enum __attribute__((mode(V8HI))) { E5 } v2; // expected-error{{mode 'V8HI' is not supported for enumeration types}}
                                              // expected-warning@-1{{deprecated}}

  // Incomplete enums without mode attribute are not allowed.
  typedef enum Y IncompleteYType; // expected-note{{forward declaration of 'enum Y'}}

  enum X a1; // expected-error{{variable has incomplete type 'enum X'}}
             // expected-note@-1{{forward declaration of 'enum X'}}
  IncompleteYType a2; // expected-error{{variable has incomplete type 'IncompleteYType' (aka 'enum Y')}}

  // OK with 'mode' attribute.
  typedef enum Y __attribute__((mode(QI))) CompleteYType1;
  typedef enum Y CompleteYType2 __attribute__((mode(HI)));
  typedef enum { A1, B1 } __attribute__((mode(QI))) CompleteType3;
  typedef enum { A2, B2 } CompleteType4 __attribute__((mode(QI)));
  typedef enum __attribute__((mode(QI))) { A3, B3 } CompleteType5;

  enum X __attribute__((mode(QI))) a3;
  enum X a4 __attribute__((mode(HI)));
  IncompleteYType __attribute__((mode(QI))) a5;
  IncompleteYType a6 __attribute__((mode(HI)));
  CompleteYType1 a7;
  CompleteYType2 a8;
  CompleteType3 a9;
  CompleteType4 a10;
  CompleteType5 a11;
  enum __attribute__((mode(QI))) { A4, B4 } a12;

  return 0;
}
