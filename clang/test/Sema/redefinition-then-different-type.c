// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -x c++ %s

// Previously crashed with an assertion in IntExprEvaluator::Success.
// https://github.com/llvm/llvm-project/issues/227733

const char e = 1; // expected-note 2 {{previous definition is here}}
const char e = 1; // expected-error {{redefinition of 'e'}}
const unsigned e; // expected-error {{redefinition of 'e' with a different type: 'const unsigned int' vs 'const char'}}
int x = e;
