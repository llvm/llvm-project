// RUN: %clang_cc1 -Wunsafe-buffer-usage -verify %s

void testIncrement(char *p) {
  ++p; // expected-warning{{unchecked operation on raw buffer in expression}}
  p++; // expected-warning{{unchecked operation on raw buffer in expression}}
  --p; // expected-warning{{unchecked operation on raw buffer in expression}}
  p--; // expected-warning{{unchecked operation on raw buffer in expression}}
}

void foo(...);   // let arguments of `foo` to hold testing expressions

void * voidPtrCall(void);
char * charPtrCall(void);

void testArraySubscripts(int *p, int **pp) {
  foo(p[0],             // expected-warning{{unchecked operation on raw buffer in expression}}
      pp[0][0],         // expected-warning2{{unchecked operation on raw buffer in expression}}
      0[0[pp]],         // expected-warning2{{unchecked operation on raw buffer in expression}}
      0[pp][0]          // expected-warning2{{unchecked operation on raw buffer in expression}}
      );

  if (p[3]) {           // expected-warning{{unchecked operation on raw buffer in expression}}
    void * q = p;

    foo(((int*)q)[10]); // expected-warning{{unchecked operation on raw buffer in expression}}
  }

  foo(((int*)voidPtrCall())[3], // expected-warning{{unchecked operation on raw buffer in expression}}
      3[(int*)voidPtrCall()],   // expected-warning{{unchecked operation on raw buffer in expression}}
      charPtrCall()[3],         // expected-warning{{unchecked operation on raw buffer in expression}}
      3[charPtrCall()]          // expected-warning{{unchecked operation on raw buffer in expression}}
      );

  int a[10], b[10][10];

  // not to warn subscripts on arrays
  foo(a[0], a[1],
      0[a], 1[a],
      b[3][4],
      4[b][3],
      4[3[b]]);
}

void testArraySubscriptsWithAuto(int *p, int **pp) {
  int a[10];

  auto ap1 = a;

  foo(ap1[0]);  // expected-warning{{unchecked operation on raw buffer in expression}}

  auto ap2 = p;

  foo(ap2[0]);  // expected-warning{{unchecked operation on raw buffer in expression}}

  auto ap3 = pp;

  foo(pp[0][0]); // expected-warning2{{unchecked operation on raw buffer in expression}}

  auto ap4 = *pp;

  foo(ap4[0]);  // expected-warning{{unchecked operation on raw buffer in expression}}
}

void testUnevaluatedContext(int * p) {
  //TODO: do not warn for unevaluated context
  foo(sizeof(p[1]),             // expected-warning{{unchecked operation on raw buffer in expression}}
      sizeof(decltype(p[1])));  // expected-warning{{unchecked operation on raw buffer in expression}}
}
