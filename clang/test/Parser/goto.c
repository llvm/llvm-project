/* RUN: %clang_cc1 -fsyntax-only -verify %s
*/

void test1(void) { 
  goto ; /* expected-error {{expected identifier}} */
}


void test2(void) {
  l:  /* expected-note {{previous definition is here}} */
  
  {
    __label__ l;
  l: goto l;
  }
  
  {
    __label__ l;
    __label__ h;   /* expected-error {{use of undeclared label 'h'}} */
  l: goto l;
  }

  /* PR3429 */
  {
  l:  /* expected-error {{redefinition of label 'l'}} */
    ;
  }

}
