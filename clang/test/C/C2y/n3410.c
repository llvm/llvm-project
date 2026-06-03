// RUN: %clang_cc1 -verify=c2y -std=c2y -Wall -pedantic -Wno-unused %s
// RUN: %clang_cc1 -verify=c89-23 -std=c23 -Wall -pedantic -Wno-unused %s
// RUN: %clang_cc1 -verify=c89-23 -std=c17 -Wall -pedantic -Wno-unused %s
// RUN: %clang_cc1 -verify=c89-23 -std=c11 -Wall -pedantic -Wno-unused %s
// RUN: %clang_cc1 -verify=c89-23 -std=c99 -Wall -pedantic -Wno-unused %s
// RUN: %clang_cc1 -verify=c89-23 -std=c89 -Wall -pedantic -Wno-unused -Wno-comment %s

/* WG14 N3410: Clang 23
 * Slay Some Earthly Demons XI
 *
 * It is now ill-formed for the same identifier within a TU to have both
 * internal and external linkage.
 */

void func1(void) {
  extern int a; /* #a */
}

/* This 'a' is the same as the one declared extern above. */
static int a; /* c2y-error {{static declaration of 'a' follows non-static declaration}}
                 c2y-note@#a {{previous declaration is here}}
                 c89-23-error {{static declaration of 'a' follows non-static declaration}}
                 c89-23-note@#a {{previous declaration is here}}
               */

static int b;
void func2(void) {
  /* This 'b' is the same as the one declared static above, but this is not
     ill-formed because of C2y 6.2.2p4, which gives this variable internal
     linkage because the previous declaration had internal linkage.
   */
  extern int b; /* Ok */
}

static int c, d; /* c2y-note 2 {{previous definition is here}}
                    c89-23-note 2 {{previous definition is here}}
                  */
void func3(void) {
  int c; /* no linkage, different object from the one declared above. */
  {
    int d; /* no linkage, different object from the file-scope 'd'. */
    {
      /* This 'c' is the same as the one declared at file scope, but because
         of the local scope 'c', the file scope 'c' is not visible. */
      extern int c; /* c2y-error {{'c' declared with both internal and external linkage in the same translation unit}}
                       c89-23-error {{'c' declared with both internal and external linkage in the same translation unit; behavior is undefined}}
                     */
      /* This 'd' is the same as the one declared at file scope as well, but
         because of the enclosing block-scope 'd', the file scope 'd' is also
         not visible, same as with 'c'. */
      extern int d; /* c2y-error {{'d' declared with both internal and external linkage in the same translation unit}}
                       c89-23-error {{'d' declared with both internal and external linkage in the same translation unit; behavior is undefined}}
                     */
    }
  }
  {
    static int e;
    {
      extern int e; /* Ok for the same reason as 'b' above. */
    }
  }
}

/* A function parameter shadows the file-scope 'p' the same way a local
   variable does, so the block-scope 'extern' does not inherit internal
   linkage and conflicts. */
static int p; /* c2y-note {{previous definition is here}}
                 c89-23-note {{previous definition is here}}
               */
void func4(int p) {
  {
    extern int p; /* c2y-error {{'p' declared with both internal and external linkage in the same translation unit}}
                     c89-23-error {{'p' declared with both internal and external linkage in the same translation unit; behavior is undefined}}
                   */
  }
}

static int q;
void func5(void) {
  /* No shadow intervenes here, so this 'q' inherits the internal linkage of
     the file-scope 'q', which is fine. */
  extern int q; /* #q */
  {
    int q; /* no linkage; shadows the declarations above. */
    {
      /* The file-scope 'q' is now hidden, so this 'extern' has external
         linkage and conflicts with the internal-linkage declaration above. */
      extern int q; /* c2y-error {{'q' declared with both internal and external linkage in the same translation unit}}
                       c2y-note@#q {{previous declaration is here}}
                       c89-23-error {{'q' declared with both internal and external linkage in the same translation unit; behavior is undefined}}
                       c89-23-note@#q {{previous declaration is here}}
                     */
    }
  }
}

void func6(void) {
  /* No file-scope declaration of 'r' exists, so the block-scope 'extern' just
     has external linkage and there is no conflict. */
  {
    int r; /* no linkage. */
    {
      extern int r; /* Ok */
    }
  }
}
