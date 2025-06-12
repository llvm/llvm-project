// Check that clang-diagnostic pragmas take precedence over suppression mapping.
// RUN: %clang -cc1 -verify -Wformat=2 --warning-suppression-mappings=%S/Inputs/suppression-mapping.txt -fsyntax-only %s

__attribute__((__format__ (__printf__, 1, 2)))
void format_err(const char* const pString, ...);

void foo() {
  const char *x;
  format_err(x); // Warning suppressed here.
  // check that pragmas take precedence
#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wformat=2"
  format_err(x); // expected-warning{{format string is not a string literal (potentially insecure)}} \
                 // expected-note{{treat the string as an argument to avoid this}}
#pragma clang diagnostic pop
}
