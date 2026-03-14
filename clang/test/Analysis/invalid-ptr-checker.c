// RUN: %clang_analyze_cc1                                                      \
// RUN:  -analyzer-checker=security.cert.env.InvalidPtr                         \
// RUN:  -analyzer-config security.cert.env.InvalidPtr:InvalidatingGetEnv=false \
// RUN:  -analyzer-output=text -verify -Wno-unused %s
//
// RUN: %clang_analyze_cc1                                                      \
// RUN:  -analyzer-checker=security.cert.env.InvalidPtr                         \
// RUN:  -analyzer-config                                                       \
// RUN: security.cert.env.InvalidPtr:InvalidatingGetEnv=true                    \
// RUN: -analyzer-output=text -verify=expected,pedantic -Wno-unused %s

#include "Inputs/system-header-simulator.h"

char *getenv(const char *name);
int setenv(const char *name, const char *value, int overwrite);
int strcmp(const char *, const char *);

int custom_env_handler(const char **envp);

void getenv_after_getenv(void) {
  char *v1 = getenv("V1");
  // pedantic-note@-1{{previous function call was here}}

  char *v2 = getenv("V2");
  // pedantic-note@-1{{'getenv' call may invalidate the result of the previous 'getenv'}}

  strcmp(v1, v2);
  // pedantic-warning@-1{{use of invalidated pointer 'v1' in a function call}}
  // pedantic-note@-2{{use of invalidated pointer 'v1' in a function call}}
}

void setenv_after_getenv(void) {
  char *v1 = getenv("VAR1");

  setenv("VAR2", "...", 1);
  // expected-note@-1{{'setenv' call may invalidate the environment returned by 'getenv'}}

  strcmp(v1, "");
  // expected-warning@-1{{use of invalidated pointer 'v1' in a function call}}
  // expected-note@-2{{use of invalidated pointer 'v1' in a function call}}
}

int main(int argc, const char *argv[], const char *envp[]) {
  setenv("VAR", "...", 0);
  // expected-note@-1 2 {{'setenv' call may invalidate the environment parameter of 'main'}}

  *envp;
  // expected-warning@-1 2 {{dereferencing an invalid pointer}}
  // expected-note@-2 2 {{dereferencing an invalid pointer}}
}

void multiple_invalidation_no_duplicate_notes(void) {
  char *v1 = getenv("VAR1");

  setenv("VAR2", "...", 1); // no note here

  setenv("VAR3", "...", 1);
  // expected-note@-1{{'setenv' call may invalidate the environment returned by 'getenv'}}

  strcmp(v1, "");
  // expected-warning@-1{{use of invalidated pointer 'v1' in a function call}}
  // expected-note@-2{{use of invalidated pointer 'v1' in a function call}}
}
