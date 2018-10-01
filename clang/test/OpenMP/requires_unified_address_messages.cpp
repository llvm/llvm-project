// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100  %s

#pragma omp requires unified_address // expected-note {{unified_address clause previously used here}} expected-note {{unified_address clause previously used here}} expected-note {{unified_address clause previously used here}} expected-note {{unified_address clause previously used here}} expected-note {{unified_address clause previously used here}} expected-note{{unified_address clause previously used here}}

#pragma omp requires unified_shared_memory // expected-note {{unified_shared_memory clause previously used here}} expected-note{{unified_shared_memory clause previously used here}}

#pragma omp requires unified_shared_memory, unified_shared_memory // expected-error {{Only one unified_shared_memory clause can appear on a requires directive in a single translation unit}} expected-error {{directive '#pragma omp requires' cannot contain more than one 'unified_shared_memory' clause}}

#pragma omp requires unified_address // expected-error {{Only one unified_address clause can appear on a requires directive in a single translation unit}} 

#pragma omp requires unified_address, unified_address // expected-error {{Only one unified_address clause can appear on a requires directive in a single translation unit}} expected-error {{directive '#pragma omp requires' cannot contain more than one 'unified_address' clause}}

#pragma omp requires // expected-error {{expected at least one clause on '#pragma omp requires' directive}}

#pragma omp requires invalid_clause // expected-warning {{extra tokens at the end of '#pragma omp requires' are ignored}} expected-error {{expected at least one clause on '#pragma omp requires' directive}}

#pragma omp requires nowait // expected-error {{unexpected OpenMP clause 'nowait' in directive '#pragma omp requires'}} expected-error {{expected at least one clause on '#pragma omp requires' directive}}

#pragma omp requires unified_address, invalid_clause // expected-warning {{extra tokens at the end of '#pragma omp requires' are ignored}} expected-error {{Only one unified_address clause can appear on a requires directive in a single translation unit}}

#pragma omp requires invalid_clause unified_address // expected-warning {{extra tokens at the end of '#pragma omp requires' are ignored}} expected-error {{expected at least one clause on '#pragma omp requires' directive}}

#pragma omp requires unified_shared_memory, unified_address // expected-error {{Only one unified_shared_memory clause can appear on a requires directive in a single translation unit}} expected-error{{Only one unified_address clause can appear on a requires directive in a single translation unit}}

namespace A {
  #pragma omp requires unified_address // expected-error {{Only one unified_address clause can appear on a requires directive in a single translation unit}}
  namespace B {
    #pragma omp requires unified_address // expected-error {{Only one unified_address clause can appear on a requires directive in a single translation unit}}
  }
}

template <typename T> T foo() {
  #pragma omp requires unified_address // expected-error {{unexpected OpenMP directive '#pragma omp requires'}}
}

class C {
  #pragma omp requires unified_address // expected-error {{'#pragma omp requires' directive must appear only in file scope}}
};

int main() {
  #pragma omp requires unified_address // expected-error {{unexpected OpenMP directive '#pragma omp requires'}}
}
