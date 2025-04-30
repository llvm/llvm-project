// RUN: %clang_cc1 -std=c++11 -verify %s

// Note that this puts the expected lines before the directives to work around
// limitations in the -verify mode.

void test(int *List, int Length, int Value) {
#pragma ns location grid
  for (int i = 0; i < Length; i++) {
    List[i] = Value;
  }

  /* expected-error {{expected a valid location in '#pragma ns location'}} */ #pragma ns location
      /* expected-error {{expected only one location in '#pragma ns location'}} */ #pragma ns location grid risc //.
      for (int i = 0; i < Length; i++) {
    for (int j = 0; j < Length; j++) {
      List[i * Length + j] = Value;
    }

#pragma ns location grid
    /* expected-error {{expected a for, while, or do-while loop to follow '#pragma ns location'}} */ i = 2;
  }
}

#pragma ns location host
/* expected-error {{expected a valid location in '#pragma ns location'}} */ #pragma ns location
    /* expected-error {{invalid ns location argument (expected {grid|risc|host})}} */ #pragma ns location a b //
    int
    func() {
  return 1;
}

#pragma ns location grid
int grid() {
  return 2;
}

#pragma ns location risc
int risc() {
  return 9;
}

/* expected-error {{expected only one location in '#pragma ns location'}} */ #pragma ns location grid risc //
int two_locations() {
  return 9;
}

#pragma ns location host
int host() {
  return 11;
}

void test_loops(int *List, int Length, int Value) {
#pragma ns location grid
  for (int i = 0; i < Length; i++) {
    List[i] = Value;
  }
}
