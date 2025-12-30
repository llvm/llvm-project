// RUN: %clang_cc1 %s -fsyntax-only -verify -pedantic

// Test that we can parse declarations at global scope.
int v;

void func(void) {
  // Test that we can parse declarations within a compound statement.
  int a;
  {
    int b;
  }

  int z = ({ // expected-warning {{use of GNU statement expression extension}}
	// Test that we can parse declarations within a GNU statement expression.
	int w = 12;
	w;
  });

  // Test that we diagnose declarations where a statement is required.
  // See GH92775.
  if (1)
    int x; // expected-error {{expected expression}}
  for (;;)
    int c; // expected-error {{expected expression}}

  label:
    int y; // expected-warning {{label followed by a declaration is a C23 extension}}

  // Test that lookup works as expected.
  (void)a;
  (void)v;
  (void)z;
  (void)b; // expected-error {{use of undeclared identifier 'b'}}
  (void)w; // expected-error {{use of undeclared identifier 'w'}}
  (void)x; // expected-error {{use of undeclared identifier 'x'}}
  (void)c; // expected-error {{use of undeclared identifier 'c'}}
  (void)y;
}

