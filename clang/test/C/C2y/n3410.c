// RUN: %clang_cc1 -verify-directives -verify=expected,c2y -std=c2y -Wall -pedantic -Wno-unused %s
// RUN: %clang_cc1 -verify-directives -verify=expected,c89-23 -std=c23 -Wall -pedantic -Wno-unused %s
// RUN: %clang_cc1 -verify-directives -verify=expected,c89-23 -std=c17 -Wall -pedantic -Wno-unused %s
// RUN: %clang_cc1 -verify-directives -verify=expected,c89-23 -std=c11 -Wall -pedantic -Wno-unused %s
// RUN: %clang_cc1 -verify-directives -verify=expected,c89-23 -std=c99 -Wall -pedantic -Wno-unused %s
// RUN: %clang_cc1 -verify-directives -verify=expected,c89-23 -std=c89 -Wall -pedantic -Wno-unused -Wno-comment %s

/* WG14 N3410: Clang 24
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
                 c89-23-error {{static declaration of 'a' follows non-static declaration; behavior is undefined}}
                 expected-note@#a {{previous declaration is here}}
               */

static int b;
void func2(void) {
  /* This 'b' is well-formed, because C2y 6.2.2p6 makes it "inherit" the
     static linkage of `static int b` above, because the latter is visible.
   */
  extern int b; /* Ok */
}

static int c, d; /* #c_d */
void func3(void) {
  int c; /* no linkage, different object from the one declared above. */
  {
    int d; /* no linkage, different object from the file-scope 'd'. */
    {
      /* This 'c' is the same as the one declared at file scope, but because
         of the local scope 'c', the file scope 'c' is not visible. */
      extern int c; /* c2y-error {{variable 'c' cannot be declared with external linkage following a declaration with internal linkage}}
                       c89-23-error {{variable 'c' declared with external linkage following a declaration with internal linkage; behavior is undefined}}
                       expected-note@#c_d {{previous declaration is here}}
                     */
      /* This 'd' is the same as the one declared at file scope as well, but
         because of the enclosing block-scope 'd', the file scope 'd' is also
         not visible, same as with 'c'. */
      extern int d; /* c2y-error {{variable 'd' cannot be declared with external linkage following a declaration with internal linkage}}
                       c89-23-error {{variable 'd' declared with external linkage following a declaration with internal linkage; behavior is undefined}}
                       expected-note@#c_d {{previous declaration is here}}
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
static int p; /* #p */
void func4(int p) {
  {
    extern int p; /* c2y-error {{variable 'p' cannot be declared with external linkage following a declaration with internal linkage}}
                     c89-23-error {{variable 'p' declared with external linkage following a declaration with internal linkage; behavior is undefined}}
                     expected-note@#p {{previous declaration is here}}
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
      extern int q; /* c2y-error {{variable 'q' cannot be declared with external linkage following a declaration with internal linkage}}
                       c89-23-error {{variable 'q' declared with external linkage following a declaration with internal linkage; behavior is undefined}}
                       expected-note@#q {{previous declaration is here}}
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

static int s; /* #s */
void func7(void) {
  {
    /* The file-scope 's' is visible here, so this 'extern' inherits its
       internal linkage, which may be surprising. */
    extern int s; /* Ok */
  }
  {
    int s; /* no linkage; shadows the file-scope 's'. */
    {
      /* The file-scope 's' is hidden by the local 's' above, so this 'extern'
         has external linkage and conflicts.

         This tests that we do not accidentally note the internal linkage
         declaration using the 'extern' specifier in the function scope; we
         want the note to point to the declaration using the 'static'
         specifier at global scope because the function scope identifier is
         hidden at this point. */
      extern int s; /* c2y-error {{variable 's' cannot be declared with external linkage following a declaration with internal linkage}}
                       c89-23-error {{variable 's' declared with external linkage following a declaration with internal linkage; behavior is undefined}}
                       expected-note@#s {{previous declaration is here}}
                     */
    }
  }
}
