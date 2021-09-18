// RUN: %clang_analyze_cc1                                         \
// RUN:  -analyzer-checker=core,alpha.security.cert.env.InvalidPtr \
// RUN:  -verify -Wno-unused %s

#include "../Inputs/system-header-simulator.h"
char *getenv(const char *name);
int strcmp(const char*, const char*);
char *strdup(const char*);
void free(void *memblock);
void *malloc(size_t size);

void incorrect_usage(void) {
  char *tmpvar;
  char *tempvar;

  tmpvar = getenv("TMP");

  if (!tmpvar)
    return;

  tempvar = getenv("TEMP");

  if (!tempvar)
    return;

  if (strcmp(tmpvar, tempvar) == 0) { // body of strcmp is unknown
    // expected-warning@-1{{use of invalidated pointer 'tmpvar' in a function call}}
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
