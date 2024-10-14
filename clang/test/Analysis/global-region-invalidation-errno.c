// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -disable-free -verify %s \
// RUN:   -analyzer-checker=core,deadcode,optin.taint \
// RUN:   -DERRNO_VAR

// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -disable-free -verify %s \
// RUN:   -analyzer-checker=core,deadcode,optin.taint \
// RUN:   -DERRNO_FUNC

// Note, we do need to include headers here, since the analyzer checks if the function declaration is located in a system header.
// The errno value can be defined in multiple ways, test with each one.
#ifdef ERRNO_VAR
#include "Inputs/errno_var.h"
#endif
#ifdef ERRNO_FUNC
#include "Inputs/errno_func.h"
#endif
#include "Inputs/system-header-simulator.h"


void foo(void);

// expected-no-diagnostics

// Test errno gets invalidated by a system call.
int testErrnoSystem(void) {
  int i;
  int *p = 0;
  fscanf(stdin, "%d", &i);
  if (errno == 0) {
    fscanf(stdin, "%d", &i); // errno gets invalidated here.
    return 5 / errno; // no-warning
  }

  errno = 0;
  fscanf(stdin, "%d", &i); // errno gets invalidated here.
  return 5 / errno; // no-warning
}

// Test that errno gets invalidated by internal calls.
int testErrnoInternal(void) {
  int i;
  int *p = 0;
  fscanf(stdin, "%d", &i);
  if (errno == 0) {
    foo(); // errno gets invalidated here.
    return 5 / errno; // no-warning
  }
  return 0;
}
