// RUN: %clang_cc1 -triple=x86_64-unknown-unknown -verify -fcf-protection=branch -fsyntax-only %s

// Function pointer definition.
typedef void (*FuncPointerWithNoCfCheck)(void) __attribute__((nocf_check)); // no-warning
typedef void (*FuncPointer)(void);

// Dont allow function declaration and definition mismatch.
void __attribute__((nocf_check)) testNoCfCheck();   // expected-note {{previous declaration is here}}
void testNoCfCheck(){}; //  expected-error {{conflicting types for 'testNoCfCheck'}}

// No variable or parameter declaration
__attribute__((nocf_check)) int i;                              // expected-warning {{'nocf_check' attribute only applies to function}}
void testNoCfCheckImpl(double __attribute__((nocf_check)) i) {} // expected-warning {{'nocf_check' attribute only applies to function}}

// Allow attributed function pointers as well as casting between attributed
// and non-attributed function pointers.
void testNoCfCheckMismatch(FuncPointer f) {
  FuncPointerWithNoCfCheck fNoCfCheck = f; // expected-warning {{incompatible function pointer types}}
  (*fNoCfCheck)();                         // no-warning
}

// 'nocf_check' Attribute has no parameters.
int testNoCfCheckParams() __attribute__((nocf_check(1))); // expected-error {{'nocf_check' attribute takes no arguments}}
