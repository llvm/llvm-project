// RUN: %clang_cc1 -std=c++11 -verify %s

// Note that this puts the expected lines before the directives to work around
// limitations in the -verify mode.

void test(int *List, int Length, int Value) {
  int i = 0;

#pragma unroll_and_jam
  for (int i = 0; i < Length; i++) {
    for (int j = 0; j < Length; j++) {
      List[i * Length + j] = Value;
    }
  }

#pragma nounroll_and_jam
  for (int i = 0; i < Length; i++) {
    for (int j = 0; j < Length; j++) {
      List[i * Length + j] = Value;
    }
  }

#pragma unroll_and_jam 4
  for (int i = 0; i < Length; i++) {
    for (int j = 0; j < Length; j++) {
      List[i * Length + j] = Value;
    }
  }

/* expected-error {{expected ')'}} */ #pragma unroll_and_jam(4
/* expected-error {{missing argument; expected an integer value}} */ #pragma unroll_and_jam()
/* expected-warning {{extra tokens at end of '#pragma unroll_and_jam'}} */ #pragma unroll_and_jam 1 2
  for (int i = 0; i < Length; i++) {
    for (int j = 0; j < Length; j++) {
      List[i * Length + j] = Value;
    }
  }

/* expected-warning {{extra tokens at end of '#pragma nounroll_and_jam'}} */ #pragma nounroll_and_jam 1
  for (int i = 0; i < Length; i++) {
    for (int j = 0; j < Length; j++) {
      List[i * Length + j] = Value;
    }
  }

#pragma unroll_and_jam
/* expected-error {{expected a for, while, or do-while loop to follow '#pragma unroll_and_jam'}} */ int j = Length;
#pragma unroll_and_jam 4
/* expected-error {{expected a for, while, or do-while loop to follow '#pragma unroll_and_jam'}} */ int k = Length;
#pragma nounroll_and_jam
/* expected-error {{expected a for, while, or do-while loop to follow '#pragma nounroll_and_jam'}} */ int l = Length;

#pragma unroll_and_jam 4
/* expected-error {{incompatible directives '#pragma nounroll_and_jam' and '#pragma unroll_and_jam(4)'}} */ #pragma nounroll_and_jam
  for (int i = 0; i < Length; i++) {
    for (int j = 0; j < Length; j++) {
      List[i * Length + j] = Value;
    }
  }

#pragma nounroll_and_jam
#pragma unroll(4)
  for (int i = 0; i < Length; i++) {
    for (int j = 0; j < Length; j++) {
      List[i * Length + j] = Value;
    }
  }

// pragma clang unroll_and_jam is disabled for the moment
/* expected-error {{invalid option 'unroll_and_jam'; expected vectorize, vectorize_width, interleave, interleave_count, unroll, unroll_count, or distribute}} */ #pragma clang loop unroll_and_jam(4)
  for (int i = 0; i < Length; i++) {
    for (int j = 0; j < Length; j++) {
      List[i * Length + j] = Value;
    }
  }

#pragma unroll_and_jam
/* expected-error {{expected statement}} */ }
