// RUN: %clang_cc1 -fsyntax-only -Wloop-analysis -verify %s

struct S {
  bool stop() { return false; }
  bool keep_running;
};

void by_ref(int &value) { }
void by_value(int value) { }
void by_pointer(int *value) {}

void test1() {
  S s;
  for (; !s.stop();) {}
  for (; s.keep_running;) {}
  for (int i; i < 1; ++i) {}
  for (int i; i < 1; ) {}  // expected-warning {{variable 'i' used in loop condition not modified in loop body}}
  for (int i; i < 1; ) { ++i; }
  for (int i; i < 1; ) { return; }
  for (int i; i < 1; ) { break; }
  for (int i; i < 1; ) { goto exit_loop; }
exit_loop:
  for (int i; i < 1; ) { by_ref(i); }
  for (int i; i < 1; ) { by_value(i); }  // expected-warning {{variable 'i' used in loop condition not modified in loop body}}
  for (int i; i < 1; ) { by_pointer(&i); }

  for (int i; i < 1; ++i)
    for (int j; j < 1; ++j)
      { }
  for (int i; i < 1; ++i)
    for (int j; j < 1; ++i)  // expected-warning {{variable 'j' used in loop condition not modified in loop body}}
      { }
  for (int i; i < 1; ++i)
    for (int j; i < 1; ++j)  // expected-warning {{variable 'i' used in loop condition not modified in loop body}}
      { }

  for (int *i, *j; i < j; ++i) {}
  for (int *i, *j; i < j;) {}  // expected-warning {{variables 'i' and 'j' used in loop condition not modified in loop body}}

  // Dereferencing pointers is ignored for now.
  for (int *i; *i; ) {}
}

void test2() {
  int i, j, k;
  int *ptr;

  // Testing CastExpr
  for (; i; ) {} // expected-warning {{variable 'i' used in loop condition not modified in loop body}}
  for (; i; ) { i = 5; }

  // Testing BinaryOperator
  for (; i < j; ) {} // expected-warning {{variables 'i' and 'j' used in loop condition not modified in loop body}}
  for (; i < j; ) { i = 5; }
  for (; i < j; ) { j = 5; }

  // Testing IntegerLiteral
  for (; i < 5; ) {} // expected-warning {{variable 'i' used in loop condition not modified in loop body}}
  for (; i < 5; ) { i = 5; }

  // Testing FloatingLiteral
  for (; i < 5.0; ) {} // expected-warning {{variable 'i' used in loop condition not modified in loop body}}
  for (; i < 5.0; ) { i = 5; }

  // Testing CharacterLiteral
  for (; i == 'a'; ) {} // expected-warning {{variable 'i' used in loop condition not modified in loop body}}
  for (; i == 'a'; ) { i = 5; }

  // Testing CXXBoolLiteralExpr
  for (; i == true; ) {} // expected-warning {{variable 'i' used in loop condition not modified in loop body}}
  for (; i == true; ) { i = 5; }

  // Testing GNUNullExpr
  for (; ptr == __null; ) {} // expected-warning {{variable 'ptr' used in loop condition not modified in loop body}}
  for (; ptr == __null; ) { ptr = &i; }

  // Testing UnaryOperator
  for (; -i > 5; ) {} // expected-warning {{variable 'i' used in loop condition not modified in loop body}}
  for (; -i > 5; ) { ++i; }

  // Testing ImaginaryLiteral
  for (; i != 3i; ) {} // expected-warning {{variable 'i' used in loop condition not modified in loop body}}
  for (; i != 3i; ) { ++i; }

  // Testing ConditionalOperator
  for (; i ? j : k; ) {} // expected-warning {{variables 'i', 'j', and 'k' used in loop condition not modified in loop body}}
  for (; i ? j : k; ) { ++i; }
  for (; i ? j : k; ) { ++j; }
  for (; i ? j : k; ) { ++k; }
  for (; i; ) { j = i ? i : i; }  // expected-warning {{variable 'i' used in loop condition not modified in loop body}}
  for (; i; ) { j = (i = 1) ? i : i; }
  for (; i; ) { j = i ? i : ++i; }

  // Testing BinaryConditionalOperator
  for (; i ?: j; ) {} // expected-warning {{variables 'i' and 'j' used in loop condition not modified in loop body}}
  for (; i ?: j; ) { ++i; }
  for (; i ?: j; ) { ++j; }
  for (; i; ) { j = i ?: i; }  // expected-warning {{variable 'i' used in loop condition not modified in loop body}}

  // Testing ParenExpr
  for (; (i); ) { }  // expected-warning {{variable 'i' used in loop condition not modified in loop body}}
  for (; (i); ) { ++i; }

  // Testing non-evaluated variables
  for (; i < sizeof(j); ) { }  // expected-warning {{variable 'i' used in loop condition not modified in loop body}}
  for (; i < sizeof(j); ) { ++j; }  // expected-warning {{variable 'i' used in loop condition not modified in loop body}}
  for (; i < sizeof(j); ) { ++i; }
}

// False positive and how to silence.
void test3() {
  int x;
  int *ptr = &x;
  for (;x<5;) { *ptr = 6; }  // expected-warning {{variable 'x' used in loop condition not modified in loop body}}

  for (;x<5;) {
    *ptr = 6;
    (void)x;
  }
}

// Check ordering and printing of variables.  Max variables is currently 4.
void test4() {
  int a, b, c, d, e, f;
  for (; a;);  // expected-warning {{variable 'a' used in loop condition not modified in loop body}}
  for (; a + b;);  // expected-warning {{variables 'a' and 'b' used in loop condition not modified in loop body}}
  for (; a + b + c;);  // expected-warning {{variables 'a', 'b', and 'c' used in loop condition not modified in loop body}}
  for (; a + b + c + d;);  // expected-warning {{variables 'a', 'b', 'c', and 'd' used in loop condition not modified in loop body}}
  for (; a + b + c + d + e;);  // expected-warning {{variables used in loop condition not modified in loop body}}
  for (; a + b + c + d + e + f;);  // expected-warning {{variables used in loop condition not modified in loop body}}
  for (; a + c + d + b;);  // expected-warning {{variables 'a', 'c', 'd', and 'b' used in loop condition not modified in loop body}}
  for (; d + c + b + a;);  // expected-warning {{variables 'd', 'c', 'b', and 'a' used in loop condition not modified in loop body}}
}

// Ensure that the warning doesn't fail when lots of variables are used
// in the conditional.
void test5() {
  for (int a; a+a+a+a+a+a+a+a+a+a;); // \
   // expected-warning {{variable 'a' used in loop condition not modified in loop body}}
  for (int a; a+a+a+a+a+a+a+a+a+a+a;); // \
   // expected-warning {{variable 'a' used in loop condition not modified in loop body}}
  for (int a; a+a+a+a+a+a+a+a+a+a+a+a+a+a+a+a+a+a+a+a;);  // \
   // expected-warning {{variable 'a' used in loop condition not modified in loop body}}
  for (int a; a+a+a+a+a+a+a+a+a+a+a+a+a+a+a+a+a+a+a+a+a+a+a+a+a+a+a+a+a+a;);//\
   // expected-warning {{variable 'a' used in loop condition not modified in loop body}}
}

// Ignore global variables and static variables.
int x6;
void test6() {
  static int y;
  for (;x6;);
  for (;y;);
}
