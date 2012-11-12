/* RUN: %clang_cc1 -std=c90 -fsyntax-only %s -verify -pedantic-errors
 */

enum { cast_hex = (long) (
      0x0p-1   /* expected-error {{hexadecimal floating constants are a C99 feature}} */
     ) };

/* PR2477 */
int test1(int a,int b) {return a//* This is a divide followed by block comment in c89 mode */
b;}

// comment accepted as extension    /* expected-error {{// comments are not allowed in this language}}

void test2() {
  const char * str =
    "sdjflksdjf lksdjf skldfjsdkljflksdjf kldsjflkdsj fldks jflsdkjfds" // expected-error{{string literal of length 845 exceeds maximum length 509 that C90 compilers are required to support}}
    "sdjflksdjf lksdjf skldfjsdkljflksdjf kldsjflkdsj fldks jflsdkjfds"
    "sdjflksdjf lksdjf skldfjsdkljflksdjf kldsjflkdsj fldks jflsdkjfds"
    "sdjflksdjf lksdjf skldfjsdkljflksdjf kldsjflkdsj fldks jflsdkjfds"
    "sdjflksdjf lksdjf skldfjsdkljflksdjf kldsjflkdsj fldks jflsdkjfds"
    "sdjflksdjf lksdjf skldfjsdkljflksdjf kldsjflkdsj fldks jflsdkjfds"
    "sdjflksdjf lksdjf skldfjsdkljflksdjf kldsjflkdsj fldks jflsdkjfds"
    "sdjflksdjf lksdjf skldfjsdkljflksdjf kldsjflkdsj fldks jflsdkjfds"
    "sdjflksdjf lksdjf skldfjsdkljflksdjf kldsjflkdsj fldks jflsdkjfds"
    "sdjflksdjf lksdjf skldfjsdkljflksdjf kldsjflkdsj fldks jflsdkjfds"
    "sdjflksdjf lksdjf skldfjsdkljflksdjf kldsjflkdsj fldks jflsdkjfds"
    "sdjflksdjf lksdjf skldfjsdkljflksdjf kldsjflkdsj fldks jflsdkjfds"
    "sdjflksdjf lksdjf skldfjsdkljflksdjf kldsjflkdsj fldks jflsdkjfds";
}

void test3() {
  (void)L"\u1234";  // expected-error {{unicode escape sequences are only valid in C99 or C++}}
  (void)L'\u1234';  // expected-error {{unicode escape sequences are only valid in C99 or C++}}
}

#define PREFIX(x) foo ## x
int test4() {
  int PREFIX(0p) = 0;
  int *p = &PREFIX(0p+1);
  return p[-1];
}
