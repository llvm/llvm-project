// A note points to the external source at present, so we have to ignore it.
// RUN: %clang_cc1 -triple spirv-unknown-vulkan-compute -fnative-half-type -finclude-default-header -fsyntax-only %s -verify -verify-ignore-unexpected=note
// All the errors are actually in the external source at present, so we have to ignore them.
// The notes point to the proper lines though.
// RUN: %clang_cc1 -triple spirv-unknown-vulkan-compute -fnative-half-type -finclude-default-header -fsyntax-only -DMTXTYPE %s -verify=mtxtype -verify-ignore-unexpected=error

#ifndef MTXTYPE
void matrix_var_dimensions(int Rows, unsigned Columns, uint16_t C) {
  // expected-note@-1 3{{declared here}}
  matrix<int, Rows, 1> m1;    // expected-error{{non-type template argument is not a constant expression}}
  // expected-note@-1{{function parameter 'Rows' with unknown value cannot be used in a constant expression}}
  matrix<int, 1, Columns> m2; // expected-error{{non-type template argument is not a constant expression}}
  // expected-note@-1{{function parameter 'Columns' with unknown value cannot be used in a constant expression}}
  matrix<int, C, C> m3;       // expected-error{{non-type template argument is not a constant expression}}
  // expected-note@-1{{function parameter 'C' with unknown value cannot be used in a constant expression}}
  matrix<int, char, 0> m8;    // expected-error{{template argument for non-type template parameter must be an expression}}

}
#else
struct S1 {};

enum TestEnum {
  A,
  B
};

void matrix_unsupported_element_type() {
  // The future-errors are not checked yet since they are predeclared and are ignored.
  matrix<S1, 1, 1> m1;       // future-error{{invalid matrix element type 'S1'}}
  // mtxtype-note@-1{{in instantiation of template type alias 'matrix' requested here}}
  matrix<bool, 1, 1> m2;     // future-error{{invalid matrix element type 'bool'}}
  // mtxtype-note@-1{{in instantiation of template type alias 'matrix' requested here}}
  matrix<TestEnum, 1, 1> m3; // future-error{{invalid matrix element type 'TestEnum'}}
  // mtxtype-note@-1{{in instantiation of template type alias 'matrix' requested here}}

  matrix<int, -1, 1> m4;      // future-error{{matrix row size too large}}
  // mtxtype-note@-1{{in instantiation of template type alias 'matrix' requested here}}
  matrix<int, 1, -1> m5;      // future-error{{matrix column size too large}}
  // mtxtype-note@-1{{in instantiation of template type alias 'matrix' requested here}}
  matrix<int, 0, 1> m6;       // future-error{{zero matrix size}}
  // mtxtype-note@-1{{in instantiation of template type alias 'matrix' requested here}}
  matrix<int, 1, 0> m7;       // future-error{{zero matrix size}}
  // mtxtype-note@-1{{in instantiation of template type alias 'matrix' requested here}}
  matrix<int, 1048576, 1> m9; // future-error{{matrix row size too large}}
  // mtxtype-note@-1{{in instantiation of template type alias 'matrix' requested here}}

}
#endif
