// Regression test for a crash in getpwnam_r and similar interceptors.
// RUN: %clangxx -O0 -g %s -o %t && %run %t

// XFAIL: mips

#include <assert.h>
#include <errno.h>
#include <pwd.h>
#include <signal.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>

int main(void) {
  struct passwd pwd;
  struct passwd *pwdres;
  char buf[10000];
  int res = getpwnam_r("no-such-user", &pwd, buf, sizeof(buf), &pwdres);
  assert(res == 0 || res == ENOENT);
  assert(pwdres == 0);
  return 0;
}
