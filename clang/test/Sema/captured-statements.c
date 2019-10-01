// RUN: %clang_cc1 -fsyntax-only -verify %s -fblocks

void test_gotos() {
  goto L1; // expected-error {{use of undeclared label 'L1'}}
  goto L3; // OK
  #pragma clang __debug captured
  {
L1:
    goto L2; // OK
L2:
    goto L3; // expected-error {{use of undeclared label 'L3'}}
  }
L3: ;
}

void test_break_continue() {
  while (1) {
    #pragma clang __debug captured
    {
      break; // expected-error {{'break' statement not in loop or switch statement}}
      continue; // expected-error {{'continue' statement not in loop statement}}
    }
  }
}

void test_return() {
  while (1) {
    #pragma clang __debug captured
    {
      return; // expected-error {{cannot return from default captured statement}}
    }
  }
}

void test_nest() {
  int x;
  #pragma clang __debug captured
  {
    int y;
    #pragma clang __debug captured
    {
      int z;
      #pragma clang __debug captured
      {
        x = z = y; // OK
      }
    }
  }
}

void test_nest_block() {
  __block int x; // expected-note {{'x' declared here}}
  int y;
  ^{
    int z;
    #pragma clang __debug captured
    {
      x = y; // expected-error{{__block variable 'x' cannot be captured in a captured statement}}
      y = z; // expected-error{{variable is not assignable (missing __block type specifier)}}
      z = y; // OK
    }
  }();

  __block int a; // expected-note 2 {{'a' declared here}}
  int b;
  #pragma clang __debug captured
  {
    int d;
    ^{
      a = b; // expected-error{{__block variable 'a' cannot be captured in a captured statement}}
      a = b; // (duplicate diagnostic suppressed)
      b = d; // OK - Consistent with block inside a lambda
    }();
  }
  #pragma clang __debug captured
  {
    __block int c;
    int d;
    ^{
      c = a; // expected-error{{__block variable 'a' cannot be captured in a captured statement}}
      c = d; // OK
      d = b; // expected-error{{variable is not assignable (missing __block type specifier)}}
    }();
  }
}
