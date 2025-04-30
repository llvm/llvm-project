// RUN: %clang_cc1 -std=c++11 -verify %s

// Note that this puts the expected lines before the directives to work around
// limitations in the -verify mode.

void test(int *List, int Length, int Value) {
  int i = 0;

#pragma ns location host
  for (int i = 0; i < Length; i++) {
    List[i] = Value;
  }

  /* expected-error {{expected a valid mark in '#pragma ns mark'}} */ #pragma ns mark
  for (int i = 0; i < Length; i++) {
    for (int j = 0; j < Length; j++) {
      List[i * Length + j] = Value;
    }
  }

  #pragma ns location host
  /* expected-error {{expected a for, while, or do-while loop to follow '#pragma ns location'}} */ i = 2;

  /* expected-error {{invalid ns mark argument (expected {noimport|handoff|boundary|import_single|import_recursive} on function)}} */ #pragma ns mark peripheral
  for (int i = 0; i < Length; i++) {
    for (int j = 0; j < Length; j++) {
      List[i * Length + j] = Value;
    }
  }
}

#pragma ns mark noimport
/* expected-error {{expected a valid mark in '#pragma ns mark'}} */ #pragma ns mark
/* expected-error {{expected only one mark in '#pragma ns mark'}} */ #pragma ns mark handoff import
int noimport() {
  return 1;
}

#pragma ns mark handoff
int handoff() {
  return 2;
}

/* expected-error {{invalid ns mark argument (expected {noimport|handoff|boundary|import_single|import_recursive} on function)}} */ #pragma ns mark live
int live() {
  return 3;
}

/* expected-error {{invalid ns mark argument (expected {noimport|handoff|boundary|import_single|import_recursive} on function)}} */ #pragma ns mark import
int import() {
  return 4;
}

void test_unroll(int *List, int Length, int Value) {
  int i = 0;

/* expected-error {{invalid ns mark argument (expected {noimport|handoff|boundary|import_single|import_recursive} on function)}} */ #pragma ns mark unroll_count(2)
  for (int i = 0; i < Length; i++) {
    List[i] = Value;
  }

    /* expected-error {{invalid ns mark argument (expected {noimport|handoff|boundary|import_single|import_recursive} on function)}} */ #pragma ns mark unroll_count
    for (int i = 0; i < Length; i++) {
      for (int j = 0; j < Length; j++) {
        List[i * Length + j] = Value;
    }
  }
}
