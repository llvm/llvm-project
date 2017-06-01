// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef enum { XX } EnumType;
struct S { int x; };

// Check enumerations. Vector modes on enum types must cause an error.
template <class T>
void CheckEnumerations() {
  // Check that non-vector 'mode' attribute is OK with enumeration types.
  typedef T __attribute__((mode(QI))) T1;
  typedef T T2 __attribute__((mode(HI)));
  typedef T __attribute__((mode(V8SI))) T3; // expected-error{{mode 'V8SI' is not supported for enumeration types}}
  // expected-warning@-1{{specifying vector types with the 'mode' attribute is deprecated}}

  typedef enum __attribute__((mode(HI))) { A4, B4 } T4;
  typedef enum { A5, B5 } __attribute__((mode(SI))) T5;
  typedef enum __attribute__((mode(V2SI))) { A6, B6 } T6; // expected-error{{mode 'V2SI' is not supported for enumeration types}}
                                                          // expected-warning@-1{{deprecated}}
  typedef enum { A7, B7 } __attribute__((mode(V2QI))) T7; // expected-error{{mode 'V2QI' is not supported for enumeration types}}
                                                          // expected-warning@-1{{deprecated}}
}

// Check that attribute applies only for integer and floating-point types.
// OK when instantiated with 'int', error with structure types, for example.
template <class T>
void CheckPrimitiveTypes() {
  typedef T __attribute__((mode(QI))) T1;    // expected-error{{mode attribute only supported for integer and floating-point types}}
  typedef T __attribute__((mode(V2SI))) VT1; // expected-error{{mode attribute only supported for integer and floating-point types}}
  // expected-warning@-1{{specifying vector types with the 'mode' attribute is deprecated}}
}

// Check that attribute supports certain modes. Check that wrong machine modes
// are NOT diagnosed twice during instantiation.
template <class T>
void CheckMachineMode() {
  typedef T __attribute__((mode(QI))) T1; // expected-error{{type of machine mode does not match type of base type}}
  typedef T __attribute__((mode(HI))) T2; // expected-error{{type of machine mode does not match type of base type}}
  typedef T __attribute__((mode(SI))) T3; // expected-error{{type of machine mode does not match type of base type}}
  typedef T __attribute__((mode(DI))) T4; // expected-error{{type of machine mode does not match type of base type}}
  typedef T __attribute__((mode(SF))) T5; // expected-error2{{type of machine mode does not match type of base type}}
  typedef T __attribute__((mode(DF))) T6; // expected-error2{{type of machine mode does not match type of base type}}
  typedef T __attribute__((mode(II))) T7; // expected-error{{unknown machine mode}}
  typedef T __attribute__((mode(12))) T8; // expected-error{{'mode' attribute requires an identifier}}
}

// Check attributes on function parameters.
template <class T1, class T2>
void CheckParameters(T1 __attribute__((mode(SI)))   paramSI,     // expected-note{{ignored: substitution failure}} expected-note-re{{not viable: no known conversion from '{{.*}}' (vector of 4 '{{.*}}' values) to 'EnumType' for 2nd argument}}
                     T1 __attribute__((mode(V4DI))) paramV4DI,   // expected-warning{{deprecated}}
                     T2 __attribute__((mode(SF)))   paramSF,
                     T2 __attribute__((mode(V4DF))) paramV4DF) { // expected-warning{{deprecated}}
}


// Check dependent structure.
template <class T>
struct TemplatedStruct {
  // Check fields.
  T __attribute__((mode(HI)))     x1;
  T __attribute__((mode(V4HI)))   x2;         // expected-error{{mode 'V4HI' is not supported for enumeration types}}
                                              // expected-warning@-1{{deprecated}}

  // Check typedefs.
  typedef T __attribute__((mode(DI)))   T1;
  typedef T __attribute__((mode(V8DI))) T2;   // expected-error{{mode 'V8DI' is not supported for enumeration types}}
                                              // expected-warning@-1{{deprecated}}

  // Check parameters.
  void f1(T __attribute__((mode(QI))) x) {}
  void f2(T __attribute__((mode(SF))) x) {}   // expected-error2{{type of machine mode does not match type of base type}}
  void f3(T __attribute__((mode(V4QI))) x) {} // expected-error{{mode 'V4QI' is not supported for enumeration types}}
                                              // expected-warning@-1{{deprecated}}

  // Check attribute on methods - it is invalid.
  __attribute__((mode(QI))) T g1() { return 0; } // expected-error{{'mode' attribute only applies to variables, enums, fields and typedefs}}
};



int main() {
  CheckEnumerations<int>();
  CheckEnumerations<EnumType>(); // expected-note{{in instantiation of}}

  CheckPrimitiveTypes<int>();
  CheckPrimitiveTypes<S>();      // expected-note{{in instantiation of}}

  // 'II' mode is unknown, no matter what we instantiate with.
  CheckMachineMode<int>();       // expected-note{{in instantiation of}}
  CheckMachineMode<EnumType>();  // expected-note{{in instantiation of}}
  CheckMachineMode<float>();     // expected-note{{in instantiation of}}

  int   __attribute__((mode(V4DI))) valV4DI; // expected-warning{{deprecated}}
  float __attribute__((mode(V4DF))) valV4DF; // expected-warning{{deprecated}}
  // OK.
  CheckParameters<int, float>(0, valV4DI, 1.0, valV4DF);
  // Enumeral type with vector mode is invalid.
  CheckParameters<EnumType, float>(0, valV4DI, 1.0, valV4DF); // expected-error{{no matching function for call}}
  // 'V4DF' mode with 'int' type is invalid.
  CheckParameters<int, int>(0, valV4DI, 1, valV4DF); // expected-error{{no matching function for call}}

  TemplatedStruct<int>      s1; // expected-note{{in instantiation of}}
  TemplatedStruct<EnumType> s2; // expected-note{{in instantiation of}}
  return 0;
}
