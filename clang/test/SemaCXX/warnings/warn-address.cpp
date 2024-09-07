// RUN: %clang_cc1 -fsyntax-only -verify -Wno-bool-conversion -Wno-string-compare -Wno-tautological-compare -Waddress %s
// RUN: %clang_cc1 -fsyntax-only -verify %s

void foo();
int arr[5];
int global;
const char* str = "";

void test() {
  if (foo) {}            // expected-warning{{always evaluate to 'true'}} \
                         // expected-note{{silence}}
  if (arr) {}            // expected-warning{{always evaluate to 'true'}}
  if (&global) {}        // expected-warning{{always evaluate to 'true'}}
  if (foo == 0) {}       // expected-warning{{always false}} \
                         // expected-note{{silence}}
  if (arr == 0) {}       // expected-warning{{always false}}
  if (&global == 0) {}   // expected-warning{{always false}}

  if (str == "foo") {}   // expected-warning{{unspecified}}
}
