// RUN: %clang_analyze_cc1 \
// RUN:  -analyzer-checker=security.PutenvWithAuto \
// RUN:  -verify %s

#include "Inputs/system-header-simulator.h"
void free(void *);
void *malloc(size_t);
int putenv(char *);
int snprintf(char *, size_t, const char *, ...);

int test_auto_var(const char *var) {
  char env[1024];
  (void)snprintf(env, sizeof(env), "TEST=%s", var);
  return putenv(env); // expected-warning{{The 'putenv' function should not be called with arguments that have automatic storage}}
}

int test_static_var(const char *var) {
  static char env[1024];
  (void)snprintf(env, sizeof(env), "TEST=%s", var);
  return putenv(env);
}

void test_heap_memory(const char *var) {
  const char *env_format = "TEST=%s";
  const size_t len = strlen(var) + strlen(env_format);
  char *env = (char *)malloc(len);
  if (env == NULL)
    return;
  if (putenv(env) != 0) // no-warning: env was dynamically allocated.
    free(env);
}

typedef struct {
  int A;
  char Env[1024];
} Mem;

int test_auto_var_struct() {
  Mem mem;
  return putenv(mem.Env); // expected-warning{{The 'putenv' function should not be called with arguments that have automatic storage}}
}

int test_auto_var_subarray() {
  char env[1024];
  return putenv(env + 100); // expected-warning{{The 'putenv' function should not be called with arguments that have automatic storage}}
}

int test_constant() {
  char *env = "TEST";
  return putenv(env);
}

extern char *ext_env;
int test_extern() {
  return putenv(ext_env); // no-warning: extern storage class.
}

void test_auto_var_reset() {
  char env[] = "NAME=value";
  // TODO: False Positive
  putenv(env); // expected-warning{{The 'putenv' function should not be called with arguments that have automatic storage}}
  /*
  ...
  */
  putenv((char *)"NAME=anothervalue");
}
