// RUN: %clang_cc1 -std=c++1z -verify -Wuninitialized %s

void testIf() {
  if (bool b; b) // expected-warning {{uninitialized}} expected-note {{to silence}}
    ;
  if (int a, b = 2; a) // expected-warning {{uninitialized}} expected-note {{to silence}}
    ;
  int a;
  if (a = 0; a) {} // OK
}

void testSwitch() {
  switch (bool b; b) { // expected-warning {{uninitialized}} expected-warning {{boolean value}} expected-note {{to silence}}
    case 0:
      break;
  }
  switch (int a, b = 7; a) { // expected-warning {{uninitialized}} expected-note {{to silence}}
    case 0:
      break;
  }
  int c;
  switch (c = 0; c) { // OK
    case 0:
      break;
  }
}
