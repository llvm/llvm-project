// RUN: %clang_cc1 -std=c++11 -verify %s

// Note that this puts the expected lines before the directives to work around
// limitations in the -verify mode.

void test(int *List, int Length) {
  int i = 0;

#pragma clang loop vectorize(assume_safety)
#pragma clang loop interleave(assume_safety)
  while (i + 1 < Length) {
    List[i] = i;
  }

/* expected-error {{expected ')'}} */ #pragma clang loop vectorize(assume_safety
/* expected-error {{expected ')'}} */ #pragma clang loop interleave(assume_safety

/* expected-error {{invalid argument; expected 'enable', 'full' or 'disable'}} */ #pragma clang loop unroll(assume_safety)
/* expected-error {{invalid argument; expected 'enable' or 'disable'}} */ #pragma clang loop distribute(assume_safety)

/* expected-error {{invalid argument; expected 'enable', 'assume_safety' or 'disable'}} */ #pragma clang loop vectorize(badidentifier)
/* expected-error {{invalid argument; expected 'enable', 'assume_safety' or 'disable'}} */ #pragma clang loop interleave(badidentifier)
/* expected-error {{invalid argument; expected 'enable', 'full' or 'disable'}} */ #pragma clang loop unroll(badidentifier)
  while (i-7 < Length) {
    List[i] = i;
  }

/* expected-error {{duplicate directives 'vectorize(assume_safety)' and 'vectorize(enable)'}} */ #pragma clang loop vectorize(enable)
#pragma clang loop vectorize(assume_safety)
/* expected-error {{duplicate directives 'interleave(assume_safety)' and 'interleave(enable)'}} */ #pragma clang loop interleave(enable)
#pragma clang loop interleave(assume_safety)
  while (i-9 < Length) {
    List[i] = i;
  }
}
