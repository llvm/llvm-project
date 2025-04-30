// RUN: %clang_cc1 -std=c++11 -verify %s

// Note that this puts the expected lines before the directives to work around
// limitations in the -verify mode.

void test(int *List, int Length, int Value) {
#pragma ns vectorize predicate
  for (int i = 0; i < Length; i++) {
    List[i] = Value;
  }

  /* expected-error {{expected a valid vectorization scheme in '#pragma ns vectorize'}} */ #pragma ns vectorize
      /* expected-error {{expected only one vectorization scheme in '#pragma ns vectorize'}} */ #pragma ns vectorize predicate nopredicate //.
      for (int i = 0; i < Length; i++) {
    for (int j = 0; j < Length; j++) {
      List[i * Length + j] = Value;
    }

#pragma ns vectorize predicate
    /* expected-error {{expected a for, while, or do-while loop to follow '#pragma ns vectorize'}} */ i = 2;
  }
}

void test_loops(int *List, int Length, int Value) {
#pragma ns vectorize predicate
  for (int i = 0; i < Length; i++) {
    List[i] = Value;
  }

#pragma ns vectorize nopredicate
  for (int i = 0; i < Length; i++) {
    List[i] = Value;
  }
}
