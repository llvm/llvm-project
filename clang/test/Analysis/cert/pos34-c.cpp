// RUN: %clang_analyze_cc1 \
// RUN:  -analyzer-checker=alpha.security.cert.pos.34c\
// RUN:  -verify %s

// Examples from the CERT rule's page.
// https://wiki.sei.cmu.edu/confluence/x/6NYxBQ

#include "../Inputs/system-header-simulator.h"
void free(void *memblock);
void *malloc(size_t size);
int putenv(char *);
int snprintf(char *str, size_t size, const char *format, ...);

namespace test_auto_var_used_bad {

int volatile_memory1(const char *var) {
  char env[1024];
  int retval = snprintf(env, sizeof(env), "TEST=%s", var);
  if (retval < 0 || (size_t)retval >= sizeof(env)) {
    /* Handle error */
  }

  return putenv(env);
  // expected-warning@-1 {{The 'putenv' function should not be called with arguments that have automatic storage}}
}

} // namespace test_auto_var_used_bad

namespace test_auto_var_used_good {

int test_static(const char *var) {
  static char env[1024];

  int retval = snprintf(env, sizeof(env), "TEST=%s", var);
  if (retval < 0 || (size_t)retval >= sizeof(env)) {
    /* Handle error */
  }

  return putenv(env);
}

int test_heap_memory(const char *var) {
  static char *oldenv;
  const char *env_format = "TEST=%s";
  const size_t len = strlen(var) + strlen(env_format);
  char *env = (char *)malloc(len);
  if (env == NULL) {
    return -1;
  }
  if (putenv(env) != 0) { // no-warning: env was dynamically allocated.
    free(env);
    return -1;
  }
  if (oldenv != NULL) {
    free(oldenv); /* avoid memory leak */
  }
  oldenv = env;
  return 0;
}

} // namespace test_auto_var_used_good
