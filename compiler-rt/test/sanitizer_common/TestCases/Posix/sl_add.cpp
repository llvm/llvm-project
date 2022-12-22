// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s
//
// UNSUPPORTED: darwin, target={{.*(linux|solaris).*}}

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <stringlist.h>

int main(void) {
  printf("sl_add\n");

  StringList *sl = sl_init();
  assert(sl);
  char *p = strdup("entry");
  assert(!sl_add(sl, p));
  char *entry = sl_find(sl, "entry");
  assert(!strcmp(entry, p));
  printf("Found '%s'\n", entry);
  sl_free(sl, 1);

  return 0;
  // CHECK: sl_add
  // CHECK: Found '{{.*}}'
}
