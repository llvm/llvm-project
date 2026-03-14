// RUN: %clang_cc1 -std=c++11 -verify %s

// Note that this puts the expected lines before the directives to work around
// limitations in the -verify mode.

void test(int *List, int Length, int Value) {
  int i = 0;

#pragma clang loop pipeline(disable)
  for (int i = 0; i < Length; i++) {
    List[i] = Value;
  }

#pragma clang loop pipeline_initiation_interval(10)
  for (int i = 0; i < Length; i++) {
    List[i] = Value;
  }

/* expected-error {{expected ')'}} */ #pragma clang loop pipeline(disable
/* expected-error {{invalid argument; expected 'disable'}} */ #pragma clang loop pipeline(enable)
/* expected-error {{invalid argument; expected 'disable'}} */ #pragma clang loop pipeline(error)
/* expected-error {{expected '('}} */ #pragma clang loop pipeline disable
/* expected-error {{missing argument; expected an integer value}} */ #pragma clang loop pipeline_initiation_interval()
/* expected-error {{use of undeclared identifier 'error'}} */ #pragma clang loop pipeline_initiation_interval(error)
/* expected-error {{expected '('}} */ #pragma clang loop pipeline_initiation_interval 1 2
/* expected-error {{expected ')'}} */ #pragma clang loop pipeline_initiation_interval(1
  for (int i = 0; i < Length; i++) {
    for (int j = 0; j < Length; j++) {
      List[i * Length + j] = Value;
    }
  }

}
