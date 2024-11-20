// Check for warnings
// RUN: not %clang --warning-suppression-mappings=foo.txt -fsyntax-only %s 2>&1 | FileCheck -check-prefix MISSING_MAPPING %s
// RUN: not %clang -cc1 --warning-suppression-mappings=foo.txt -fsyntax-only %s 2>&1 | FileCheck -check-prefix MISSING_MAPPING %s
// MISSING_MAPPING: error: no such file or directory: 'foo.txt'

// Check that it's no-op when diagnostics aren't enabled.
// RUN: %clang -cc1 -Wno-everything -Werror --warning-suppression-mappings=%S/Inputs/suppression-mapping.txt -fsyntax-only %s 2>&1 | FileCheck -check-prefix WARNINGS_DISABLED --allow-empty %s
// WARNINGS_DISABLED-NOT: warning:
// WARNINGS_DISABLED-NOT: error:

// RUN: %clang -cc1 -verify -Wformat=2 -Wunused --warning-suppression-mappings=%S/Inputs/suppression-mapping.txt -fsyntax-only %s

__attribute__((__format__ (__printf__, 1, 2)))
void format_err(const char* const pString, ...);

namespace {
void foo() {
  const char *x;
  format_err(x); // Warning suppressed here.
  const char *y; // expected-warning{{unused variable 'y'}}
}
}

#line 42 "foo/bar.h"
namespace {
void bar() { // expected-warning{{unused function 'bar'}}
  const char *x;
  format_err(x); // expected-warning{{format string is not a string literal (potentially insecure)}} \
                 // expected-note{{treat the string as an argument to avoid this}}
}
}
