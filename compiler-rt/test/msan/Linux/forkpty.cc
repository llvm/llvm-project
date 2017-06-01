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

  char ttyname[255];
  ttyname_r(master, ttyname, sizeof(ttyname));
  assert(__msan_test_shadow(ttyname, strlen(ttyname) + 1) == -1);

  int master2;
  forkpty(&master2, NULL, NULL, NULL);
  assert(__msan_test_shadow(&master2, sizeof(master2)) == -1);
}
