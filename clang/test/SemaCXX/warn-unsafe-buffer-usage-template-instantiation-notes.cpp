// Without -fsafe-buffer-usage-suggestions, -Wunsafe-buffer-usage analysis
// runs per-Decl during parsing, so the template instantiation stack is
// available and instantiation notes are emitted (rdar://107480207).
//
// With -fsafe-buffer-usage-suggestions, the analysis runs at end-of-TU
// where the instantiation stack is no longer available, so instantiation
// notes are not yet emitted.


// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage -verify %s
// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage -fsafe-buffer-usage-suggestions \
// RUN:            -verify=with-fixit %s

void use(int);

template<typename T>
void templateFunction(T *p) { // with-fixit-warning{{'p' is an unsafe pointer used for buffer access}}
  use(p[5]); // expected-warning{{unsafe buffer access}} \
                expected-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}} \
                with-fixit-note{{used in buffer access here}}
}

void instantiate(int *p) {
  templateFunction(p); // expected-note{{in instantiation of function template specialization 'templateFunction<int>' requested here}}
}

template<typename T>
T templateReturn(T *p) { // with-fixit-warning{{'p' is an unsafe pointer used for buffer access}}
  return p[5]; // expected-warning{{unsafe buffer access}} \
                  expected-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}} \
                  with-fixit-note{{used in buffer access here}}
}

void instantiate2(int *p) {
  templateReturn(p); // expected-note{{in instantiation of function template specialization 'templateReturn<int>' requested here}}
}

template<typename T>
void templateArithmetic(T *p) { // with-fixit-warning{{'p' is an unsafe pointer used for buffer access}}
  use(*(p + 3)); // expected-warning{{unsafe pointer arithmetic}} \
                    expected-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}} \
                    with-fixit-note{{used in pointer arithmetic here}}
}

void instantiate3(int *p) {
  templateArithmetic(p); // expected-note{{in instantiation of function template specialization 'templateArithmetic<int>' requested here}}
}
