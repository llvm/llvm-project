// RUN: %clang_cc1 %s -verify -fopenacc

template<unsigned I, typename T>
void templ() {
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc loop collapse(I)
  for(;;){}

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc loop collapse(T::value)
  for(;;){}
}

struct S {
  static constexpr unsigned value = 5;
};

void use() {
  templ<7, S>();
}
