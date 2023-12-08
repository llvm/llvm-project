// RUN: %clang_analyze_cc1 \
// RUN:  -analyzer-checker=alpha.security.cert.pos.34c\
// RUN:  -verify %s

#include "../Inputs/system-header-simulator.h"
void free(void *memblock);
void *malloc(size_t size);
int putenv(char *);
int rand();

namespace test_auto_var_used_good {

extern char *ex;
int test_extern() {
  return putenv(ex); // no-warning: extern storage class.
}

void foo(void) {
  char *buffer = (char *)"huttah!";
  if (rand() % 2 == 0) {
    buffer = (char *)malloc(5);
    strcpy(buffer, "woot");
  }
  putenv(buffer);
}

void bar(void) {
  char *buffer = (char *)malloc(5);
  strcpy(buffer, "woot");

  if (rand() % 2 == 0) {
    free(buffer);
    buffer = (char *)"blah blah blah";
  }
  putenv(buffer);
}

void baz() {
  char env[] = "NAME=value";
  // TODO: False Positive
  putenv(env);
  // expected-warning@-1 {{The 'putenv' function should not be called with arguments that have automatic storage}}

  /*
    DO SOMETHING
  */

  putenv((char *)"NAME=anothervalue");
}

} // namespace test_auto_var_used_good
