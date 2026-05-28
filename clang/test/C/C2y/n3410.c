// RUN: %clang_cc1 -verify=c2y -std=c2y -Wall -pedantic -Wno-unused %s
// RUN: %clang_cc1 -verify=pre -std=c23 -Wall -pedantic -Wno-unused %s
// RUN: %clang_cc1 -verify=pre -std=c17 -Wall -pedantic -Wno-unused %s
// RUN: %clang_cc1 -verify=pre -std=c11 -Wall -pedantic -Wno-unused %s
// RUN: %clang_cc1 -verify=pre -std=c99 -Wall -pedantic -Wno-unused %s
// RUN: %clang_cc1 -verify=pre -std=c89 -Wall -pedantic -Wno-unused -Wno-comment %s

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
                 pre-error {{static declaration of 'a' follows non-static declaration}}
                 pre-note@#a {{previous declaration is here}}
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
                    pre-note 2 {{previous definition is here}}
                  */
void func3(void) {
  int c; /* no linkage, different object from the one declared above. */
  {
    int d; /* no linkage, different object from the file-scope 'd'. */
    {
      /* This 'c' is the same as the one declared at file scope, but because
         of the local scope 'c', the file scope 'c' is not visible. */
      extern int c; /* c2y-error {{'c' declared with both internal and external linkage in the same translation unit}}
                       pre-error {{'c' declared with both internal and external linkage in the same translation unit; behavior is undefined}}
                     */
      /* This 'd' is the same as the one declared at file scope as well, but
         because of the enclosing block-scope 'd', the file scope 'd' is also
         not visible, same as with 'c'. */
      extern int d; /* c2y-error {{'d' declared with both internal and external linkage in the same translation unit}}
                       pre-error {{'d' declared with both internal and external linkage in the same translation unit; behavior is undefined}}
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

