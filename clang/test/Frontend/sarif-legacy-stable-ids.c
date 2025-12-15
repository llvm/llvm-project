// RUN: %clang_cc1 -triple avr-unknown-unknown -fsyntax-only -fdiagnostics-format sarif %s > %t 2>&1
// RUN: cat %t | %normalize_sarif | diff -U1 -b %S/Inputs/expected-sarif/sarif-legacy-stable-ids.c.sarif -

struct a { int b; };

// Warning 'warn_interrupt_signal_attribute_invalid' was previously known as 'warn_interrupt_attribute_invalid'.
// In SARIF, it will be referred to by its new ID, with a "deprecatedIds" entry specifying the old ID.
__attribute__((interrupt)) int fooa(void) { return 0; }
