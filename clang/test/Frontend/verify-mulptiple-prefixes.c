// Test the diagnostic messages of -verify with multiple prefixes.
// - Expected but not seen errors should contain the prefix of the directive
// - Seen but not expected errors should not choose an arbitrary prefix
// - "expected directive cannot follow '<prefix>-no-diagnostics'" should report an actual
//    expected-no-diagnostics prefix present in the source.

// RUN: not %clang_cc1 -verify=foo,bar %s 2>&1 | FileCheck %s --check-prefix=CHECK1
// RUN: not %clang_cc1 -verify=bar,foo %s 2>&1 | FileCheck %s --check-prefix=CHECK1

undefined_type x; // #1

// foo-error{{there is no error here}}
// bar-error{{error not seen}}
// bar-note{{declared here}}
// bar-error{{another error not seen}}
// bar-error-re{{regex error{{}} not present}}

// CHECK1: error: diagnostics with 'error' severity expected but not seen: 
// CHECK1:   Line 12 'foo-error': there is no error here
// CHECK1:   Line 13 'bar-error': error not seen
// CHECK1:   Line 15 'bar-error': another error not seen
// CHECK1:   Line 16 'bar-error-re': regex error{{{{[}][}]}} not present
// CHECK1: error: diagnostics with 'error' severity seen but not expected: 
// CHECK1:   Line 10: unknown type name 'undefined_type'
// CHECK1: error: diagnostics with 'note' severity expected but not seen: 
// CHECK1:   Line 14 'bar-note': declared here
// CHECK1: 6 errors generated.

// RUN: not %clang_cc1 -verify=baz,qux,quux %s 2>&1 | FileCheck %s --check-prefix=CHECK2

// qux-no-diagnostics
// baz-error@#1{{unknown type name 'undefined_type'}}
// quux-no-diagnostics
// qux-error-re@#1{{unknown type name 'undefined_type'}}

// CHECK2: error: diagnostics with 'error' severity seen but not expected: 
// CHECK2:   Line 10: unknown type name 'undefined_type'
// CHECK2:   Line 32: 'baz-error' directive cannot follow 'qux-no-diagnostics' directive
// CHECK2:   Line 34: 'qux-error-re' directive cannot follow 'qux-no-diagnostics' directive

// RUN: not %clang_cc1 -verify=spam,eggs %s 2>&1 | FileCheck %s --check-prefix=CHECK3

// eggs-error@#1{{unknown type name 'undefined_type'}}
// spam-no-diagnostics

// CHECK3: error: diagnostics with 'error' severity seen but not expected: 
// CHECK3:   Line 44: 'spam-no-diagnostics' directive cannot follow other expected directives
// CHECK3: 1 error generated.
