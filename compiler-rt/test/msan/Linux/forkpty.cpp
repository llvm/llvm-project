// RUN: %clangxx_msan -O0 -g %s -lutil -o %t && %run %t

#include <assert.h>
#include <pty.h>
#include <unistd.h>
#include <cstring>

#include <sanitizer/msan_interface.h>

int
main (int argc, char** argv)
{
  int parent, worker;
  openpty(&parent, &worker, NULL, NULL, NULL);
  assert(__msan_test_shadow(&parent, sizeof(parent)) == -1);
  assert(__msan_test_shadow(&worker, sizeof(worker)) == -1);

  char name[255];
  ttyname_r(parent, name, sizeof(name));
  assert(__msan_test_shadow(name, strlen(name) + 1) == -1);

  char *name_p = ttyname(parent);
  assert(__msan_test_shadow(name_p, strlen(name_p) + 1) == -1);

  int parent2;
  forkpty(&parent2, NULL, NULL, NULL);
  assert(__msan_test_shadow(&parent2, sizeof(parent2)) == -1);
}
