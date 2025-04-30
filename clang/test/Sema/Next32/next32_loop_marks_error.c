// // RUN: %clang_cc1 %s -fsyntax-only -verify
// // RUN: %clang_cc1 %s -triple=x86_64-unknown-linux-gnu -fsyntax-only -verify

void test_slot_noarg(int *List, int Length, int Value) {
#pragma ns mark slot // expected-error {{expected '('}}
  for (int i = 0; i < Length; i++) {
    List[i] = Value;
  }
}

void test_slot_no_string(int *List, int Length, int Value) {
#pragma ns mark slot(123) // expected-error {{invalid argument of type 'int'; expected a string literal}}
  for (int i = 0; i < Length; i++) {
    List[i] = Value;
  }
}

void test_cgid_noarg(int *List, int Length, int Value) {
#pragma ns mark cgid // expected-error {{expected '('}}
  for (int i = 0; i < Length; i++) {
    List[i] = Value;
  }
}

void test_cgid_no_string(int *List, int Length, int Value) {
#pragma ns mark cgid(6) // expected-error {{invalid argument of type 'int'; expected a string literal}}
  for (int i = 0; i < Length; i++) {
    List[i] = Value;
  }
}
