// RUN: %clang_analyze_cc1 \
// RUN:  -analyzer-checker=security.PutenvStackArray \
// RUN:  -verify %s

#include "Inputs/system-header-simulator.h"
void free(void *);
void *malloc(size_t);
int putenv(char *);
int snprintf(char *, size_t, const char *, ...);

int test_auto_var(const char *var) {
  char env[1024];
  (void)snprintf(env, sizeof(env), "TEST=%s", var);
  return putenv(env); // expected-warning{{The 'putenv' function should not be called with arrays that have automatic storage}}
}

int test_static_var(const char *var) {
  static char env[1024];
  (void)snprintf(env, sizeof(env), "TEST=%s", var);
  return putenv(env); // no-warning: static array is used
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
  return putenv(mem.Env); // expected-warning{{The 'putenv' function should not be called with}}
}

int test_auto_var_subarray() {
  char env[1024];
  return putenv(env + 100); // expected-warning{{The 'putenv' function should not be called with}}
}

int f_test_auto_var_call(char *env) {
  return putenv(env); // expected-warning{{The 'putenv' function should not be called with}}
}

int test_auto_var_call() {
  char env[1024];
  return f_test_auto_var_call(env);
}

int test_constant() {
  char *env = "TEST";
  return putenv(env); // no-warning: data is not on the stack
}

extern char *ext_env;
int test_extern() {
  return putenv(ext_env); // no-warning: extern storage class.
}

void test_auto_var_reset() {
  char env[] = "NAME=value";
  putenv(env); // expected-warning{{The 'putenv' function should not be called with}}
  // ... (do something)
  // Even cases like this are likely a bug:
  // It looks like that if one string was passed to putenv,
  // it should not be deallocated at all, because when reading the
  // environment variable a pointer into this string is returned.
  // In this case, if another (or the same) thread reads variable "NAME"
  // at this point and does not copy the returned string, the data may
  // become invalid.
  putenv((char *)"NAME=anothervalue");
}

void f_main(char *env) {
  putenv(env); // no warning: string allocated in stack of 'main'
}

int main(int argc, char **argv) {
  char env[] = "NAME=value";
  putenv(env); // no warning: string allocated in stack of 'main'
  f_main(env);
  return 0;
}
