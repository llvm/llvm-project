/* RUN: %clang_cc1 -fsyntax-only %s -verify
 */

typedef void Void;

void foo(void) {
  int X;
  
  X = sizeof(int (void a));    // expected-error {{argument may not have 'void' type}}
  X = sizeof(int (int, void)); // expected-error {{must be the first and only parameter}}
  X = sizeof(int (void, ...)); // expected-error {{must be the first and only parameter}}

  X = sizeof(int (Void a));    // expected-error {{argument may not have 'void' type}}
  X = sizeof(int (int, Void)); // expected-error {{must be the first and only parameter}}
  X = sizeof(int (Void, ...)); // expected-error {{must be the first and only parameter}}

  // Accept these.
  X = sizeof(int (void));
  X = sizeof(int (Void));
}

// this is ok.
void bar(Void) {
}

void f(const void);            // expected-error {{parameter must not have type qualifiers}}
