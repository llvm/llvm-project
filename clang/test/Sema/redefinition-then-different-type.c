// RUN: %clang_cc1 -fsyntax-only -verify -std=c23 %s
// RUN: %clang_cc1 -fsyntax-only -verify -fexperimental-new-constant-interpreter -std=c23 %s
// RUN: %clang_cc1 -fsyntax-only -verify -x c++ %s
// RUN: %clang_cc1 -fsyntax-only -verify -fexperimental-new-constant-interpreter -x c++ %s

// Previously crashed with an assertion in IntExprEvaluator::Success.
// https://github.com/llvm/llvm-project/issues/227733

constexpr char e = 1; // expected-note 2 {{previous definition is here}}
constexpr char e = 1; // expected-error {{redefinition of 'e'}}
constexpr unsigned e; // expected-error {{redefinition of 'e' with a different type: 'const unsigned int' vs 'const char'}}
int x = e;

// Test when all previous declarations in the redeclaration chain are invalid:
// nothing should crash and the redefinition chain should still compile cleanly.
invalid_type_1 a; // expected-error {{unknown type name 'invalid_type_1'}}
invalid_type_2 a; // expected-error {{unknown type name 'invalid_type_2'}}
extern int a;
void test_a(void) { int y = a; }

// Test: valid declaration, followed by an invalid one, followed by a valid one,
// followed by a use.
extern int g;
invalid_type_3 g; // expected-error {{unknown type name 'invalid_type_3'}}
extern int g;
void test_g(void) { int z = g; }
