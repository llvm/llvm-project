// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

void a(int i);
int b();
int c();

#define MACRO_A 0

void test1(int x, int y) {
  while(true) {
    if (x); // expected-warning {{if statement has empty body}} expected-note{{put the semicolon on a separate line to silence this warning}}

    // Check that we handle conditions that start or end with a macro
    // correctly.
    if (x == MACRO_A); // expected-warning {{if statement has empty body}} expected-note{{put the semicolon on a separate line to silence this warning}}
    if (MACRO_A == x); // expected-warning {{if statement has empty body}} expected-note{{put the semicolon on a separate line to silence this warning}}

    int i;
    // PR11329
    for (i = 0; i < x; i++); { // expected-warning{{for loop has empty body}} expected-note{{put the semicolon on a separate line to silence this warning}}
      a(i);
      b();
    }

    for (i = 0; i < x; i++); // expected-warning{{for loop has empty body}} expected-note{{put the semicolon on a separate line to silence this warning}}
    {
      a(i);
    }

    for (i = 0;
         i < x;
         i++); // expected-warning{{for loop has empty body}} expected-note{{put the semicolon on a separate line to silence this warning}}
    {
      a(i);
    }

    int arr[3] = { 1, 2, 3 };
    for (int j : arr); // expected-warning{{range-based for loop has empty body}} expected-note{{put the semicolon on a separate line to silence this warning}}
      a(i);

    for (int j :
         arr); // expected-warning{{range-based for loop has empty body}} expected-note{{put the semicolon on a separate line to silence this warning}}
      a(i);

    while (b() == 0); // expected-warning{{while loop has empty body}} expected-note{{put the semicolon on a separate line to silence this warning}}
      a(i);

    while (b() == 0); { // expected-warning{{while loop has empty body}} expected-note{{put the semicolon on a separate line to silence this warning}}
      a(i);
    }

    while (b() == 0); // expected-warning{{while loop has empty body}} expected-note{{put the semicolon on a separate line to silence this warning}}
    {
      a(i);
    }

    while (b() == 0 ||
           c() == 0); // expected-warning{{while loop has empty body}} expected-note{{put the semicolon on a separate line to silence this warning}}
    {
      a(i);
    }

    do;          // expected-note{{to match this 'do'}}
      b();       // expected-error{{expected 'while' in do/while loop}}
    while (b()); // no-warning
    c();

    do;          // expected-note{{to match this 'do'}}
      b();       // expected-error{{expected 'while' in do/while loop}}
    while (b()); // expected-warning{{while loop has empty body}} expected-note{{put the semicolon on a separate line to silence this warning}}
      c();

    switch(x) // no-warning
    {
      switch(y); // expected-warning{{switch statement has empty body}} expected-note{{put the semicolon on a separate line to silence this warning}}
      {
        case 0:
          a(10);
          break;
        default:
          a(20);
          break;
      }
    }
  }
}

/// There should be no warning  when null statement is placed on its own line.
void test2(int x, int y) {
  if (x) // no-warning
    ; // no-warning

  int i;
  for (i = 0; i < x; i++) // no-warning
    ; // no-warning

  for (i = 0;
       i < x;
       i++) // no-warning
    ; // no-warning

  int arr[3] = { 1, 2, 3 };
  for (int j : arr) // no-warning
    ; // no-warning

  while (b() == 0) // no-warning
    ; // no-warning

  while (b() == 0 ||
         c() == 0) // no-warning
    ; // no-warning

  switch(x)
  {
    switch(y) // no-warning
      ; // no-warning
  }

  // Last `for' or `while' statement in compound statement shouldn't warn.
  while(b() == 0); // no-warning
}

/// There should be no warning for a null statement resulting from an empty macro.
#define EMPTY(a)
void test3(int x, int y) {
  if (x) EMPTY(x); // no-warning

  int i;
  for (i = 0; i < x; i++) EMPTY(i); // no-warning

  for (i = 0;
       i < x;
       i++) EMPTY(i); // no-warning

  int arr[3] = { 1, 2, 3 };
  for (int j : arr) EMPTY(j); // no-warning

  for (int j :
       arr) EMPTY(j); // no-warning

  while (b() == 0) EMPTY(i); // no-warning

  while (b() == 0 ||
         c() == 0) EMPTY(i); // no-warning

  switch (x) {
    switch (y)
      EMPTY(i); // no-warning
  }
}

void test4(int x)
{
  // Idiom used in some metaprogramming constructs.
  switch (x) default:; // no-warning

  // Frequent idiom used in macros.
  do {} while (false); // no-warning
}

/// There should be no warning for a common for/while idiom when it is obvious
/// from indentation that next statement wasn't meant to be a body.
void test5(int x, int y) {
  int i;
  for (i = 0; i < x; i++); // expected-warning{{for loop has empty body}} expected-note{{put the semicolon on a separate line to silence this warning}}
    a(i);

  for (i = 0; i < x; i++); // no-warning
  a(i);

  for (i = 0;
       i < x;
       i++); // expected-warning{{for loop has empty body}} expected-note{{put the semicolon on a separate line to silence this warning}}
    a(i);

  for (i = 0;
       i < x;
       i++); // no-warning
  a(i);

  while (b() == 0); // expected-warning{{while loop has empty body}} expected-note{{put the semicolon on a separate line to silence this warning}}
    a(i);

  while (b() == 0); // no-warning
  a(i);

  while (b() == 0 ||
         c() == 0); // expected-warning{{while loop has empty body}} expected-note{{put the semicolon on a separate line to silence this warning}}
    a(i);

  while (b() == 0 ||
         c() == 0); // no-warning
  a(i);
}

/// There should be no warning for a statement with a non-null body.
void test6(int x, int y) {
  if (x) {} // no-warning

  if (x)
    a(x); // no-warning

  int i;
  for (i = 0; i < x; i++) // no-warning
    a(i); // no-warning

  for (i = 0; i < x; i++) { // no-warning
    a(i); // no-warning
  }

  for (i = 0;
       i < x;
       i++) // no-warning
    a(i); // no-warning

  int arr[3] = { 1, 2, 3 };
  for (int j : arr) // no-warning
    a(j);

  for (int j : arr) {} // no-warning

  while (b() == 0) // no-warning
    a(i); // no-warning

  while (b() == 0) {} // no-warning

  switch(x) // no-warning
  {
    switch(y) // no-warning
    {
      case 0:
        a(10);
        break;
      default:
        a(20);
        break;
    }
  }
}

void test_if_else(int x) {
  if (x); // expected-warning{{if statement has empty body}} expected-note{{separate line}}

  if (x)
    ; // no-warning

  if (x)
    ; // no-warning
  else
    ; // no-warning

  if (x)
    ; // no-warning
  else; // expected-warning{{else clause has empty body}} expected-note{{separate line}}

  if (x)
    ; // no-warning
  else EMPTY(x); // no-warning
}

void test_errors(int x) {
  if (1)
    aa; // expected-error{{use of undeclared identifier}}
        // no empty body warning.

  int i;
  for (i = 0; i < x; i++)
    bb; // expected-error{{use of undeclared identifier}}

  int arr[3] = { 1, 2, 3 };
  for (int j : arr)
    cc; // expected-error{{use of undeclared identifier}}

  while (b() == 0)
    dd; // expected-error{{use of undeclared identifier}}
}

// Warnings for statements in templates shouldn't be duplicated for all
// instantiations.
template <typename T>
void test_template(int x) {
  if (x); // expected-warning{{if statement has empty body}} expected-note{{put the semicolon on a separate line to silence this warning}}

  if (x)
    EMPTY(x); // no-warning

  int arr[3] = { 1, 2, 3 };
  for (int j : arr); // expected-warning{{range-based for loop has empty body}} expected-note{{put the semicolon on a separate line to silence this warning}}

  while (b() == 0); // expected-warning{{while loop has empty body}} expected-note{{put the semicolon on a separate line to silence this warning}}
    a(x);
}

void test_template_inst(int x) {
  test_template<int>(x);
  test_template<double>(x);
}

#define IDENTITY(a) a
void test7(int x, int y) {
  if (x) IDENTITY(); // no-warning
}

