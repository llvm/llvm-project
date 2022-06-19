// RUN: %clangxx -O0 -g %s -o %t
//
// REQUIRES: freebsd, netbsd

#include <assert.h>
#include <stdlib.h>
#include <ttyent.h>

#include <assert.h>
#include <stdlib.h>
#include <ttyent.h>

void test1() {
  struct ttyent *typ = getttyent();
  assert(typ && typ->ty_name != nullptr);
  assert(typ->ty_type != nullptr);
  endttyent();
}

void test2() {
  struct ttyent *typ = getttynam("console");
  assert(typ && typ->ty_name != nullptr);
  assert(typ->ty_type != nullptr);
  endttyent();
}

void test3() {
  if (!setttyent())
    exit(1);

  struct ttyent *typ = getttyent();
  assert(typ && typ->ty_name != nullptr);
  assert(typ->ty_type != nullptr);
  endttyent();
}

#if defined(__NetBSD__)
void test4() {
  if (!setttyentpath(_PATH_TTYS))
    exit(1);

  struct ttyent *typ = getttyent();
  assert(typ && typ->ty_name != nullptr);
  assert(typ->ty_type != nullptr);
  assert(typ->ty_class != nullptr);

  endttyent();
}
#endif

int main(void) {
  test1();
  test2();
  test3();
#if defined(__NetBSD__)
  test4();
#endif

  return 0;
}
