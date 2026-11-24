// Check that collapse/ordered nest counts wait on instantiation-dependent
// expressions, not only value-dependent ones.
// sizeof(sizeof(T() + T())) names T, so it is instantiation-dependent, but
// its value is already sizeof(size_t).
//
// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fopenmp %s
// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fopenmp-simd %s

template <typename T>
void collapse_too_few() {
#pragma omp for collapse(sizeof(sizeof(T() + T()))) // expected-note {{as specified in 'collapse' clause}}
  for (int i = 0; i < 4; ++i)
    ; // expected-error {{expected 8 for loops after '#pragma omp for', but found only 1}}
}

template <typename T>
void ordered_too_few() {
#pragma omp for ordered(sizeof(sizeof(T() + T()))) // expected-note {{as specified in 'ordered' clause}}
  for (int i = 0; i < 4; ++i)
    ; // expected-error {{expected 8 for loops after '#pragma omp for', but found only 1}}
}

// A literal count is not instantiation-dependent. Diagnose in the template.
template <typename T>
void collapse_literal_too_few() {
#pragma omp for collapse(2) // expected-note {{as specified in 'collapse' clause}}
  for (int i = 0; i < 4; ++i)
    ; // expected-error {{expected 2 for loops after '#pragma omp for', but found only 1}}
}

template <typename T>
void collapse_enough() {
#pragma omp for collapse(sizeof(sizeof(T() + T())))
  for (int i0 = 0; i0 < 2; ++i0)
    for (int i1 = 0; i1 < 2; ++i1)
      for (int i2 = 0; i2 < 2; ++i2)
        for (int i3 = 0; i3 < 2; ++i3)
          for (int i4 = 0; i4 < 2; ++i4)
            for (int i5 = 0; i5 < 2; ++i5)
              for (int i6 = 0; i6 < 2; ++i6)
                for (int i7 = 0; i7 < 2; ++i7)
                  ;
}

void instantiate() {
  collapse_too_few<int>();
  // expected-note@-1 {{in instantiation of function template specialization 'collapse_too_few<int>' requested here}}
  ordered_too_few<int>();
  // expected-note@-1 {{in instantiation of function template specialization 'ordered_too_few<int>' requested here}}
  collapse_literal_too_few<int>();
  collapse_enough<int>();
}
