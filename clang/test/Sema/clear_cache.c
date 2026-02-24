// RUN: %clang_cc1 -fsyntax-only -verify %s

// Ensure that __builtin___clear_cache has the expected signature. Clang used
// to have a signature accepting char * while GCC had a signature accepting
// void * that was documented incorrectly.
void test(void) {
 int n = 0;
  __builtin___clear_cache(&n, &n + 1); // Ok
  
  __builtin___clear_cache((const void *)&n, (const void *)(&n + 1)); // expected-warning 2 {{passing 'const void *' to parameter of type 'void *' discards qualifiers}}
}

