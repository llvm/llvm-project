// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>
#include <ttyent.h>

#define STRING_OR_NULL(x) ((x) ? (x) : "null")

void test1() {
  struct ttyent *typ = getttyent();

  printf("%s %s %s %d %s %s %s\n", STRING_OR_NULL(typ->ty_name),
         STRING_OR_NULL(typ->ty_getty), STRING_OR_NULL(typ->ty_type),
         typ->ty_status, STRING_OR_NULL(typ->ty_window),
         STRING_OR_NULL(typ->ty_comment), STRING_OR_NULL(typ->ty_class));

  endttyent();
}

void test2() {
  struct ttyent *typ = getttynam("console");

  printf("%s %s %s %d %s %s %s\n", STRING_OR_NULL(typ->ty_name),
         STRING_OR_NULL(typ->ty_getty), STRING_OR_NULL(typ->ty_type),
         typ->ty_status, STRING_OR_NULL(typ->ty_window),
         STRING_OR_NULL(typ->ty_comment), STRING_OR_NULL(typ->ty_class));

  endttyent();
}

void test3() {
  if (!setttyent())
    exit(1);

  struct ttyent *typ = getttyent();

  printf("%s %s %s %d %s %s %s\n", STRING_OR_NULL(typ->ty_name),
         STRING_OR_NULL(typ->ty_getty), STRING_OR_NULL(typ->ty_type),
         typ->ty_status, STRING_OR_NULL(typ->ty_window),
         STRING_OR_NULL(typ->ty_comment), STRING_OR_NULL(typ->ty_class));

  endttyent();
}

void test4() {
  if (!setttyentpath(_PATH_TTYS))
    exit(1);

  struct ttyent *typ = getttyent();

  printf("%s %s %s %d %s %s %s\n", STRING_OR_NULL(typ->ty_name),
         STRING_OR_NULL(typ->ty_getty), STRING_OR_NULL(typ->ty_type),
         typ->ty_status, STRING_OR_NULL(typ->ty_window),
         STRING_OR_NULL(typ->ty_comment), STRING_OR_NULL(typ->ty_class));

  endttyent();
}

int main(void) {
  printf("ttyent\n");

  test1();
  test2();
  test3();
  test4();

  // CHECK: ttyent

  return 0;
}
