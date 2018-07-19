// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=alpha.clone.CloneChecker -analyzer-config alpha.clone.CloneChecker:MinimumCloneComplexity=10 -verify %s

// Tests that the complexity value of a macro expansion is about the same as
// the complexity value of a normal function call and the macro body doesn't
// influence the complexity. See the CloneSignature class in CloneDetection.h
// for more information about complexity values of clones.

#define MACRO_FOO(a, b) a > b ? -a * a : -b * b;

// First, manually apply MACRO_FOO and see if the code gets detected as a clone.
// This confirms that with the current configuration the macro body would be
// considered large enough to pass the MinimumCloneComplexity constraint.

int manualMacro(int a, int b) { // expected-warning{{Duplicate code detected}}
  return a > b ? -a * a : -b * b;
}

int manualMacroClone(int a, int b) { // expected-note{{Similar code here}}
  return a > b ? -a * a : -b * b;
}

// Now we actually use the macro to generate the same AST as above. They
// shouldn't be reported because the macros only slighly increase the complexity
// value and the resulting code will never pass the MinimumCloneComplexity
// constraint.

int macro(int a, int b) {
  return MACRO_FOO(a, b);
}

int macroClone(int a, int b) {
  return MACRO_FOO(a, b);
}

// So far we only tested that macros increase the complexity by a lesser amount
// than normal code. We also need to be sure this amount is not zero because
// we otherwise macro code would be 'invisible' for the CloneDetector.
// This tests that it is possible to increase the reach the minimum complexity
// by only using macros. This is only possible if the complexity value is bigger
// than zero.

#define NEG(A) -(A)

int nestedMacros() { // expected-warning{{Duplicate code detected}}
  return NEG(NEG(NEG(NEG(NEG(NEG(NEG(NEG(NEG(NEG(1))))))))));
}

int nestedMacrosClone() { // expected-note{{Similar code here}}
  return NEG(NEG(NEG(NEG(NEG(NEG(NEG(NEG(NEG(NEG(1))))))))));
}
