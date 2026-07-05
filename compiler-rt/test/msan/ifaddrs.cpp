// RUN: %clangxx_msan -O0 %s -o %t && %run %t %p 2>&1
// RUN: %clangxx_msan -O0 -D_FILE_OFFSET_BITS=64 %s -o %t && %run %t %p 2>&1
// RUN: %clangxx_msan -O3 %s -o %t && %run %t %p 2>&1

#include <assert.h>
#include <errno.h>
#include <ifaddrs.h>
#include <stdio.h>
#include <string.h>

#include <vector>

#if defined(__FreeBSD__) || defined(__NetBSD__)
#include <sys/socket.h>  // To define 'struct sockaddr'.
#endif

#include <sanitizer/msan_interface.h>

#define CHECK_AND_PUSH(addr, size)                                             \
  if (addr) {                                                                  \
    assert(-1 == __msan_test_shadow(addr, (size_t)(size)));                    \
    ranges.push_back(std::make_pair((void *)addr, (size_t)size));              \
  }

int main(int argc, char *argv[]) {
  struct ifaddrs *ifas;

  assert(0 == __msan_test_shadow(&ifas, sizeof(ifaddrs *)));
  int res = getifaddrs(&ifas);
  if (res == -1) {
    assert(errno == ENOSYS);
    printf("getifaddrs() is not implemented\n");
    return 0;
  }
  assert(res == 0);
  assert(-1 == __msan_test_shadow(&ifas, sizeof(ifaddrs *)));

  std::vector<std::pair<void *, size_t>> ranges;
  ifaddrs *p = ifas;
  while (p) {
    CHECK_AND_PUSH(p, sizeof(ifaddrs));
    CHECK_AND_PUSH(p->ifa_name, strlen(p->ifa_name) + 1);
    CHECK_AND_PUSH(p->ifa_addr, sizeof(*p->ifa_addr));
    CHECK_AND_PUSH(p->ifa_netmask, sizeof(*p->ifa_netmask));
    CHECK_AND_PUSH(p->ifa_broadaddr, sizeof(*p->ifa_broadaddr));
    CHECK_AND_PUSH(p->ifa_dstaddr, sizeof(*p->ifa_dstaddr));
    p = p->ifa_next;
  }

  freeifaddrs(ifas);
  for (int i = 0; i < ranges.size(); i++)
    assert(0 == __msan_test_shadow(ranges[i].first, ranges[i].second));
  return 0;
}
