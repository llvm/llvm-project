// RUN: %clang_cc1 -analyze -analyzer-checker=alpha.security.taint,core -analyzer-output=text -verify %s

// This file is for testing enhanced diagnostics produced by the GenericTaintChecker

int scanf(const char *restrict format, ...);
int system(const char *command);

void taintDiagnostic()
{
  char buf[128];
  scanf("%s", buf); // expected-note {{Taint originated here}}
  system(buf); // expected-warning {{Untrusted data is passed to a system call}} // expected-note {{Untrusted data is passed to a system call (CERT/STR02-C. Sanitize data passed to complex subsystems)}}
}
