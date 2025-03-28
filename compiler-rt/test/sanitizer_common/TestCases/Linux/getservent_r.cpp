// RUN: %clangxx -O0 %s -o %t && %run %t

// REQUIRES: glibc

#include <arpa/inet.h>
#include <assert.h>
#include <netdb.h>

int main(void) {
  struct servent result_buf;
  struct servent *result;
  char buf[1024];
  assert(getservent_r(&result_buf, buf, sizeof(buf), &result) == 0);
  assert(result != nullptr);

  // If these fail, check /etc/services if "ssh" exists. I picked this because
  // it should exist everywhere, if it doesn't, I am sorry. Disable the test
  // then please.
  assert(getservbyname_r("ssh", nullptr, &result_buf, buf, sizeof(buf),
                         &result) == 0);
  assert(result != nullptr);
  assert(getservbyport_r(htons(22), nullptr, &result_buf, buf, sizeof(buf),
                         &result) == 0);
  assert(result != nullptr);

  assert(getservbyname_r("invalidhadfuiasdhi", nullptr, &result_buf, buf,
                         sizeof(buf), &result) == 0);
  assert(result == nullptr);
}
