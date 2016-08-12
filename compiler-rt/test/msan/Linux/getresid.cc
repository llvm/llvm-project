// RUN: %clangxx_msan -O0 %s -o %t && %run %t %p 2>&1
// RUN: %clangxx_msan -O0 -D_FILE_OFFSET_BITS=64 %s -o %t && %run %t %p 2>&1
// RUN: %clangxx_msan -O3 %s -o %t && %run %t %p 2>&1

// XFAIL: target-is-mips64el

#include <assert.h>
#include <unistd.h>

#include <sanitizer/msan_interface.h>

int main(int argc, char *argv[]) {
  uid_t uids[6];
  assert(0 == __msan_test_shadow(uids, 6 * sizeof(uid_t)));
  assert(0 == getresuid(&uids[0], &uids[2], &uids[4]));
  for (int i = 0; i < 3; i++)
    assert(sizeof(uid_t) ==
           __msan_test_shadow(uids + 2 * i, 2 * sizeof(uid_t)));

  gid_t gids[6];
  assert(0 == __msan_test_shadow(gids, 6 * sizeof(gid_t)));
  assert(0 == getresgid(&gids[0], &gids[2], &gids[4]));
  for (int i = 0; i < 3; i++)
    assert(sizeof(gid_t) ==
           __msan_test_shadow(gids + 2 * i, 2 * sizeof(gid_t)));
  return 0;
}
