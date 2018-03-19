// RUN: %clang_cc1 -triple=i386-unknown-unknown -verify -fcf-protection=branch -target-feature +ibt -std=c++11 -fsyntax-only %s

// Function pointer definition.
[[gnu::nocf_check]] typedef void (*FuncPointerWithNoCfCheck)(void); // no-warning
typedef void (*FuncPointer)(void);

// Dont allow function declaration and definition mismatch.
[[gnu::nocf_check]] void testNoCfCheck();   // expected-note {{previous declaration is here}}
void testNoCfCheck(){}; //  expected-error {{conflicting types for 'testNoCfCheck'}}

// No variable or parameter declaration
int [[gnu::nocf_check]] i;                              // expected-error {{'nocf_check' attribute cannot be applied to types}}
void testNoCfCheckImpl(double i [[gnu::nocf_check]]) {} // expected-warning {{'nocf_check' attribute only applies to functions and function pointers}}

// Allow attributed function pointers as well as casting between attributed
// and non-attributed function pointers.
void testNoCfCheckMismatch(FuncPointer f) {
  FuncPointerWithNoCfCheck fNoCfCheck = f; // expected-error {{cannot initialize a variable of type}}
  (*fNoCfCheck)();                         // no-warning
}

// 'nocf_check' Attribute has no parameters.
[[gnu::nocf_check(1)]] int testNoCfCheckParams(); // expected-error {{'nocf_check' attribute takes no arguments}}
