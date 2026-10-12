// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -x c++ %s
// RUN: %clang_cc1 -fsyntax-only -verify -fexperimental-new-constant-interpreter %s
// RUN: %clang_cc1 -fsyntax-only -verify -fexperimental-new-constant-interpreter -x c++ %s

// Previously crashed with an assertion in IntExprEvaluator::Success.
// https://github.com/llvm/llvm-project/issues/227733

const char e = 1; // expected-note 2 {{previous definition is here}}
const char e = 1; // expected-error {{redefinition of 'e'}}
const unsigned e; // expected-error {{redefinition of 'e' with a different type: 'const unsigned int' vs 'const char'}}
int x = e;

// Test when all previous declarations in the redeclaration chain are invalid.
invalid_type_1 a; // expected-error {{unknown type name 'invalid_type_1'}}
invalid_type_2 a; // expected-error {{unknown type name 'invalid_type_2'}}
int a;
int y = a;
