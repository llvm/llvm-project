// RUN: %clangxx_msan -O0 -g %s -lutil -o %t && %run %t

#include <assert.h>
#include <pty.h>
#include <unistd.h>
#include <cstring>

#include <sanitizer/msan_interface.h>

int
main (int argc, char** argv)
{
  int master, slave;
  openpty(&master, &slave, NULL, NULL, NULL);
  assert(__msan_test_shadow(&master, sizeof(master)) == -1);
  assert(__msan_test_shadow(&slave, sizeof(slave)) == -1);

  char name[255];
  ttyname_r(master, name, sizeof(name));
  assert(__msan_test_shadow(name, strlen(name) + 1) == -1);

  char *name_p = ttyname(master);
  assert(__msan_test_shadow(name_p, strlen(name_p) + 1) == -1);

  int master2;
  forkpty(&master2, NULL, NULL, NULL);
  assert(__msan_test_shadow(&master2, sizeof(master2)) == -1);
}
