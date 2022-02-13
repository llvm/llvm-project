// RUN: %clang_cc1 -analyze -analyzer-checker=alpha.security.taint,core,alpha.security.ArrayBoundV2 -analyzer-output=text -verify %s

// This file is for testing enhanced diagnostics produced by the GenericTaintChecker

int scanf(const char *restrict format, ...);
int system(const char *command);

void taintDiagnostic(void)
{
  char buf[128];
  scanf("%s", buf); // expected-note {{Taint originated here}}
  system(buf); // expected-warning {{Untrusted data is passed to a system call}} // expected-note {{Untrusted data is passed to a system call (CERT/STR02-C. Sanitize data passed to complex subsystems)}}
}

int taintDiagnosticOutOfBound(void) {
  int index;
  int Array[] = {1, 2, 3, 4, 5};
  scanf("%d", &index); // expected-note {{Taint originated here}}
  return Array[index]; // expected-warning {{Out of bound memory access (index is tainted)}}
                       // expected-note@-1 {{Out of bound memory access (index is tainted)}}
}

int taintDiagnosticDivZero(int operand) {
  scanf("%d", &operand); // expected-note {{Value assigned to 'operand'}}
                         // expected-note@-1 {{Taint originated here}}
  return 10 / operand; // expected-warning {{Division by a tainted value, possibly zero}}
                       // expected-note@-1 {{Division by a tainted value, possibly zero}}
}

void taintDiagnosticVLA(void) {
  int x;
  scanf("%d", &x); // expected-note {{Value assigned to 'x'}}
                   // expected-note@-1 {{Taint originated here}}
  int vla[x]; // expected-warning {{Declared variable-length array (VLA) has tainted size}}
              // expected-note@-1 {{Declared variable-length array (VLA) has tainted size}}
}
