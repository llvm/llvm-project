// Default options.
// RUN: %clang_analyze_cc1                                                      \
// RUN:  -analyzer-checker=core,security.cert.env.InvalidPtr                    \
// RUN:  -verify -Wno-unused %s
//
// Test the laxer handling of getenv function (this is the default).
// RUN: %clang_analyze_cc1                                                      \
// RUN:  -analyzer-checker=core,security.cert.env.InvalidPtr                    \
// RUN:  -analyzer-config security.cert.env.InvalidPtr:InvalidatingGetEnv=false \
// RUN:  -verify -Wno-unused %s
//
// Test the stricter handling of getenv function.
// RUN: %clang_analyze_cc1                                                      \
// RUN:  -analyzer-checker=core,security.cert.env.InvalidPtr                    \
// RUN:  -analyzer-config security.cert.env.InvalidPtr:InvalidatingGetEnv=true  \
// RUN:  -verify=expected,pedantic -Wno-unused %s

#include "../Inputs/system-header-simulator.h"
char *getenv(const char *name);
int setenv(const char *name, const char *value, int overwrite);
int strcmp(const char*, const char*);
char *strdup(const char*);
void free(void *memblock);
void *malloc(size_t size);

void incorrect_usage_setenv_getenv_invalidation(void) {
  char *tmpvar;
  char *tempvar;

  tmpvar = getenv("TMP");

  if (!tmpvar)
    return;

  setenv("TEMP", "", 1); //setenv can invalidate env

  if (!tmpvar)
    return;

  if (strcmp(tmpvar, "") == 0) { // body of strcmp is unknown
    // expected-warning@-1{{use of invalidated pointer 'tmpvar' in a function call}}
  }
}

void incorrect_usage_double_getenv_invalidation(void) {
  char *tmpvar;
  char *tempvar;

  tmpvar = getenv("TMP");

  if (!tmpvar)
    return;

  tempvar = getenv("TEMP"); //getenv should not invalidate env in non-pedantic mode

  if (!tempvar)
    return;

  if (strcmp(tmpvar, tempvar) == 0) { // body of strcmp is unknown
    // pedantic-warning@-1{{use of invalidated pointer 'tmpvar' in a function call}}
  }
}

void correct_usage_1(void) {
  char *tmpvar;
  char *tempvar;

  const char *temp = getenv("TMP");
  if (temp != NULL) {
    tmpvar = (char *)malloc(strlen(temp)+1);
    if (tmpvar != NULL) {
      strcpy(tmpvar, temp);
    } else {
      return;
    }
  } else {
    return;
  }

  temp = getenv("TEMP");
  if (temp != NULL) {
    tempvar = (char *)malloc(strlen(temp)+1);
    if (tempvar != NULL) {
      strcpy(tempvar, temp);
    } else {
      return;
    }
  } else {
    return;
  }

  if (strcmp(tmpvar, tempvar) == 0) {
    printf("TMP and TEMP are the same.\n");
  } else {
    printf("TMP and TEMP are NOT the same.\n");
  }
  free(tmpvar);
  free(tempvar);
}

void correct_usage_2(void) {
  char *tmpvar;
  char *tempvar;

  const char *temp = getenv("TMP");
  if (temp != NULL) {
    tmpvar = strdup(temp);
    if (tmpvar == NULL) {
      return;
    }
  } else {
    return;
  }

  temp = getenv("TEMP");
  if (temp != NULL) {
    tempvar = strdup(temp);
    if (tempvar == NULL) {
      return;
    }
  } else {
    return;
  }

  if (strcmp(tmpvar, tempvar) == 0) {
    printf("TMP and TEMP are the same.\n");
  } else {
    printf("TMP and TEMP are NOT the same.\n");
  }
  free(tmpvar);
  tmpvar = NULL;
  free(tempvar);
  tempvar = NULL;
}
