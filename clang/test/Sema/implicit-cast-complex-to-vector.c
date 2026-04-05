// RUN: %clang_cc1 -fsyntax-only -verify=c %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify=cxx %s

typedef char __attribute__((__vector_size__(64))) V;

void implicit_cast_complex_to_vector() {
  _Complex double x;
  V y;
  // c-error@+2 {{implicit conversion from '_Complex double' to incompatible type 'V' (vector of 64 'char' values)}}
  // cxx-error@+1 {{implicit conversion from '_Complex double' to 'V' (vector of 64 'char' values) is not permitted in C++}}
  V z = x + y;
}

