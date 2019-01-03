// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s

#include <sys/capsicum.h>
#include <sys/ioctl.h>

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>

void test_cap_ioctls() {
  cap_rights_t rights;
  unsigned long ncmds[] = {TIOCGETA, TIOCGWINSZ, FIODTYPE};
  unsigned long rcmds = 0;
  cap_rights_t *rptr = cap_rights_init(&rights, CAP_IOCTL, CAP_READ);
  assert(rptr);

  int rv = cap_rights_limit(STDIN_FILENO, &rights);
  assert(rv == 0);
  rv = cap_ioctls_limit(STDIN_FILENO, ncmds, 3);
  assert(rv == 0);
  ssize_t rz = cap_ioctls_get(STDIN_FILENO, &rcmds, 3);
  assert(rz == 3);
  printf("ioctls test: %ld commands authorized\n", rz);
}

void test_cap_rights() {
  cap_rights_t rights, little, remove, grights;
  cap_rights_t *rptr = cap_rights_init(&rights, CAP_IOCTL, CAP_READ);
  assert(rptr);
  cap_rights_t *gptr = cap_rights_init(&remove, CAP_IOCTL);
  assert(gptr);
  cap_rights_t *sptr = cap_rights_init(&little, CAP_READ);
  assert(sptr);
  bool hasit = cap_rights_contains(rptr, sptr);
  assert(hasit == true);
  cap_rights_t *pptr = cap_rights_remove(&rights, gptr);
  hasit = cap_rights_contains(pptr, sptr);
  assert(hasit == true);
  cap_rights_t *aptr = cap_rights_merge(&rights, gptr);
  assert(aptr);
  bool correct = cap_rights_is_valid(&rights);
  assert(correct == true);

  int rv = cap_rights_limit(STDIN_FILENO, &rights);
  assert(rv == 0);
  rv = cap_rights_get(STDIN_FILENO, &grights);
  assert(rv == 0);
  assert(memcmp(&grights, &rights, sizeof(grights)) == 0);
  cap_rights_t *iptr = cap_rights_set(&rights, CAP_IOCTL);
  assert(iptr);
  cap_rights_t *eptr = cap_rights_clear(&rights, CAP_READ);
  assert(eptr);
  hasit = cap_rights_is_set(&rights, CAP_IOCTL);
  assert(hasit == true);
  printf("rights test: %d\n", rv);
}

int main(void) {
  test_cap_ioctls();

  test_cap_rights();

  // CHECK: ioctls test: {{.*}} commands authorized
  // CHECK: rights test: {{.*}}
}
