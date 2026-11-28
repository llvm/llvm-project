// RUN: %clang_cc1 -E -include %S/Inputs/has-include-next-absolute/test_header.h \
// RUN:   -verify %s

// Test that __has_include_next returns false when the current file was found
// via absolute path (not through the search directories). Previously, this
// would incorrectly search from the start of the include path, which could
// cause false positives or fatal errors when it tried to open non-existent
// files.

// expected-warning@Inputs/has-include-next-absolute/test_header.h:6 {{#include_next in file found relative to primary source file or found by absolute path; will search from start of include path}}

// Verify the header was included correctly
#ifndef TEST_HEADER_INCLUDED
#error "test_header.h was not included"
#endif

int main(void) { return 0; }
