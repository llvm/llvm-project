// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

#pragma clang loop pipeline(disable) /* expected-error {{expected unqualified-id}} */
int main() {
  for (int i = 0; i < 10; ++i)
    ;
}

void test(int *List, int Length, int Value) {
  int i = 0;

/* expected-error {{invalid argument of type 'double'; expected an integer type}} */ #pragma clang loop pipeline_initiation_interval(1.0)
/* expected-error {{invalid value '0'; must be positive}} */ #pragma clang loop pipeline_initiation_interval(0)
/* expected-error {{invalid value '-1'; must be positive}} */ #pragma clang loop pipeline_initiation_interval(-1)
  for (int i = 0; i < Length; i++) {
    for (int j = 0; j < Length; j++) {
      List[i * Length + j] = Value;
    }
  }

#pragma clang loop pipeline(disable) 
/* expected-error {{expected a for, while, or do-while loop to follow '#pragma clang loop'}} */ int j = Length;
#pragma clang loop pipeline_initiation_interval(4)
/* expected-error {{expected a for, while, or do-while loop to follow '#pragma clang loop'}} */ int k = Length;

#pragma clang loop pipeline(disable)
#pragma clang loop pipeline_initiation_interval(4) /* expected-error {{incompatible directives 'pipeline(disable)' and 'pipeline_initiation_interval(4)'}} */
  for (int i = 0; i < Length; i++) {
    List[i] = Value;
  }

#pragma clang loop pipeline(disable)
/* expected-error {{expected statement}} */ }

