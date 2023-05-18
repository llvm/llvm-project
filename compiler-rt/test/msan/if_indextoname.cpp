// RUN: %clangxx_msan -O0 %s -o %t && %run %t 2>&1
// RUN: %clangxx_msan -O0 -D_FILE_OFFSET_BITS=64 %s -o %t && %run %t 2>&1
// RUN: %clangxx_msan -O3 %s -o %t && %run %t 2>&1

#include <assert.h>
#include <errno.h>
#include <net/if.h>
#include <stdio.h>
#include <string.h>

#include <sanitizer/msan_interface.h>

int main(int argc, char *argv[]) {
  char ifname[IF_NAMESIZE + 1];
  assert(0 == __msan_test_shadow(ifname, sizeof(ifname)));
  if (!if_indextoname(1, ifname)) {
    assert(errno == ENXIO);
    printf("No network interfaces found.\n");
    return 0;
  }
  assert(strlen(ifname) + 1 <= __msan_test_shadow(ifname, sizeof(ifname)));
  return 0;
}
