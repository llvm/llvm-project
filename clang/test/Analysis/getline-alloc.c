// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix,debug.ExprInspection -verify %s

// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix,alpha.unix,debug.ExprInspection -verify %s

#include "Inputs/system-header-simulator.h"
#include "Inputs/system-header-simulator-for-malloc.h"

void test_getline_null_buffer() {
  FILE *F1 = tmpfile();
  if (!F1)
    return;
  char *buffer = NULL;
  size_t n = 0;
  if (getline(&buffer, &n, F1) > 0) {
    char c = buffer[0]; // ok
  }
  free(buffer);
  fclose(F1);
}

void test_getline_malloc_buffer() {
  FILE *F1 = tmpfile();
  if (!F1)
    return;

  size_t n = 10;
  char *buffer = malloc(n);
  char *ptr = buffer;

  ssize_t r = getdelim(&buffer, &n, '\r', F1);
  // ptr may be dangling
  free(ptr);    // expected-warning {{Attempt to free released memory}}
  free(buffer); // ok
  fclose(F1);
}

void test_getline_alloca() {
  FILE *F1 = tmpfile();
  if (!F1)
    return;
  size_t n = 10;
  char *buffer = alloca(n);
  getline(&buffer, &n, F1); // expected-warning {{Memory allocated by 'alloca()' should not be deallocated}}
  fclose(F1);
}

void test_getline_invalid_ptr() {
  FILE *F1 = tmpfile();
  if (!F1)
    return;
  size_t n = 10;
  char *buffer = (char*)test_getline_invalid_ptr;
  getline(&buffer, &n, F1); // expected-warning {{Argument to 'getline()' is the address of the function 'test_getline_invalid_ptr', which is not memory allocated by 'malloc()'}}
  fclose(F1);
}

void test_getline_leak() {
  FILE *F1 = tmpfile();
  if (!F1)
    return;

  char *buffer = NULL;
  size_t n = 0;
  ssize_t read;

  while ((read = getline(&buffer, &n, F1)) != -1) {
    printf("%s\n", buffer);
  }

  fclose(F1); // expected-warning {{Potential memory leak}}
}

void test_getline_stack() {
  size_t n = 10;
  char buffer[10];
  char *ptr = buffer;

  FILE *F1 = tmpfile();
  if (!F1)
    return;

  getline(&ptr, &n, F1); // expected-warning {{Argument to 'getline()' is the address of the local variable 'buffer', which is not memory allocated by 'malloc()'}}
}

void test_getline_static() {
  static size_t n = 10;
  static char buffer[10];
  char *ptr = buffer;

  FILE *F1 = tmpfile();
  if (!F1)
    return;

  getline(&ptr, &n, F1); // expected-warning {{Argument to 'getline()' is the address of the static variable 'buffer', which is not memory allocated by 'malloc()'}}
}
