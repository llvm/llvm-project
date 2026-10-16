// RUN: %clang_cc1 -fsyntax-only -verify -Werror=unreachable-code-aggressive %s

// Test that analysis-based warnings honor #pragma diagnostic controls.

struct [[clang::consumable(unconsumed)]] Linear {
  [[clang::return_typestate(unconsumed)]]
  Linear() {}
  [[clang::callable_when(consumed)]]
  ~Linear() {}
};

int a() {	
  Linear l;
  return 0; // No -Wconsumed diagnostic, analysis is not enabled.
  return 1; // expected-error {{'return' will never be executed}}
}

#pragma clang diagnostic push
#pragma clang diagnostic error "-Wconsumed"
int b() {
  Linear l;
  return 0;  // expected-error {{invalid invocation of method '~Linear' on object 'l' while it is in the 'unconsumed' state}}
  return 1;  // expected-error {{'return' will never be executed}}
}
#pragma clang diagnostic pop

int c() {
#pragma clang diagnostic push
#pragma clang diagnostic error "-Wconsumed"
  Linear l;
  return 0; // expected-error {{invalid invocation of method '~Linear' on object 'l' while it is in the 'unconsumed' state}}
  return 1; // expected-error {{'return' will never be executed}}
#pragma clang diagnostic pop
}

int d() {
#pragma clang diagnostic push
#pragma clang diagnostic error "-Wconsumed"
#pragma clang diagnostic ignored "-Wunreachable-code-aggressive"
  Linear l;
  return 0; // expected-error {{invalid invocation of method '~Linear' on object 'l' while it is in the 'unconsumed' state}}
  return 1; // Diagnostic is ignored
}
#pragma clang diagnostic pop

int e() {
#pragma clang diagnostic push
#pragma clang diagnostic error "-Wconsumed"
#pragma clang diagnostic ignored "-Wunreachable-code-aggressive"
  Linear l;
  return 0; // expected-error {{invalid invocation of method '~Linear' on object 'l' while it is in the 'unconsumed' state}}
  return 1; // Diagnostic is ignored
#pragma clang diagnostic pop
}

int f() {
  Linear l;
  return 0; // No -Wconsumed diagnostic, analysis is not enabled
  return 1; // expected-error {{'return' will never be executed}}
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunreachable-code-aggressive"
}
#pragma clang diagnostic pop	

int g() {
  Linear l;
  return 0; // No -Wconsumed diagnostic, the diagnostic generated at } is not enabled on this line.
  return 1; // expected-error {{'return' will never be executed}}
#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wconsumed"
}
#pragma clang diagnostic pop

int h() {
#pragma clang diagnostic push
#pragma clang diagnostic error "-Wconsumed"
#pragma clang diagnostic ignored "-Wunreachable-code-aggressive"
#pragma clang diagnostic pop

  Linear l;
  return 0; // No -Wconsumed diagnostic, the diagnostic generated at } is not enabled on this line.
  return 1; // expected-error {{'return' will never be executed}}
}
