/* RUN: %clang_cc1 %s -fsyntax-only -pedantic -verify -std=c89
 */

/* Top level extension marker. */

__extension__ typedef struct
{
    long long int quot; 
    long long int rem; 
} lldiv_t;


/* Decl/expr __extension__ marker. */
void bar(void) {
  __extension__ int i;
  int j;
  __extension__ (j = 10LL);
  __extension__ j = 10LL; /* expected-warning {{'long long' is an extension}} */
}

