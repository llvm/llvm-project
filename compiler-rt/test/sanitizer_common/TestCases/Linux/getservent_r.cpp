// RUN: %clangxx -O0 %s -o %t && %run %t

// REQUIRES: glibc

#include <arpa/inet.h>
#include <assert.h>
#include <errno.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void CheckResult(int ret) {
  if (ret != 0) {
    fprintf(stderr, "%s\n", strerror(errno));
    abort();
  }
}

int main(void) {
  struct servent result_buf;
  struct servent *result;
  char buf[1024];
  CheckResult(getservent_r(&result_buf, buf, sizeof(buf), &result));
  assert(result != nullptr);

  // If these fail, check /etc/services if "ssh" exists. I picked this because
  // it should exist everywhere, if it doesn't, I am sorry. Disable the test
  // then please.
  CheckResult(
      getservbyname_r("ssh", nullptr, &result_buf, buf, sizeof(buf), &result));
  assert(result != nullptr);
  CheckResult(getservbyport_r(htons(22), nullptr, &result_buf, buf, sizeof(buf),
                              &result));
  assert(result != nullptr);

  CheckResult(getservbyname_r("invalidhadfuiasdhi", nullptr, &result_buf, buf,
                              sizeof(buf), &result));
  assert(result == nullptr);
}
