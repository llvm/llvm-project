// RUN: %clang_cc1 -verify -std=c2y -Wall -pedantic -Wno-unused %s

/* WG14 N3410: No
 * Slay Some Earthly Demons XI
 *
 * It is now ill-formed for the same identifier within a TU to have both
 * internal and external linkage.
 */

void func1() {
  extern int a; // #a
}

// This 'a' is the same as the one declared extern above.
static int a; /* expected-error {{static declaration of 'a' follows non-static declaration}}
                 expected-note@#a {{previous declaration is here}}
               */

static int b;
void func2() {
  // This 'b' is the same as the one declaraed static above, but this is not
  // ill-formed because of C2y 6.2.2p4, which gives this variable internal
  // linkage because the previous declaration had internal linkage.
  extern int b; // Ok
}

static int c, d;
void func3() {
  int c; // no linkage, different object from the one declared above.
  for (int d;;) {
    // This 'c' is the same as the one declared at file scope, but because of
    // the local scope 'c', the file scope 'c' is not visible.
    // FIXME: This should be diagnosed under N3410.
    extern int c;
    // This 'd' is the same as the one declared at file scope as well, but
    // because of the 'd' declared within the for loop, the file scope 'd' is
    // also not visible, same as with 'c'.
    // FIXME: This should be diagnosed under N3410.
    extern int d;
  }
  for (static int e;;) {
    extern int e; // Ok for the same reason as 'b' above.
  }
}

