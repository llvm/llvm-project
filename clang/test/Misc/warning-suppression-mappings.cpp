// Check for warnings
// RUN: not %clang --warning-suppression-mappings=foo.txt -fsyntax-only %s 2>&1 | FileCheck -check-prefix MISSING_MAPPING %s
// RUN: not %clang -cc1 --warning-suppression-mappings=foo.txt -fsyntax-only %s 2>&1 | FileCheck -check-prefix MISSING_MAPPING %s
// MISSING_MAPPING: error: no such file or directory: 'foo.txt'

// Check that it's no-op when diagnostics aren't enabled.
// RUN: %clang -cc1 -Werror --warning-suppression-mappings=%S/Inputs/suppression-mapping.txt -fsyntax-only %s 2>&1 | FileCheck -check-prefix WARNINGS_DISABLED --allow-empty %s
// WARNINGS_DISABLED-NOT: warning:
// WARNINGS_DISABLED-NOT: error:

// RUN: %clang -cc1 -verify -Wunused --warning-suppression-mappings=%S/Inputs/suppression-mapping.txt -fsyntax-only %s

namespace { void foo(); }

#line 42 "foo/bar.h"
namespace { void bar(); } // expected-warning{{unused function 'bar'}}
