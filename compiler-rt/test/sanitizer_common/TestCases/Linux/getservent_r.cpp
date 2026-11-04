// RUN: %clangxx -O0 %s -o %t && %run %t

// REQUIRES: glibc, netbase

#include <arpa/inet.h>
#include <assert.h>
#include <fcntl.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void CheckResult(const char *file, int line, int ret) {
  if (ret != 0) {
    fprintf(stderr, "ERROR: %s:%d - %s\n", file, line, strerror(ret));
  }
  assert(ret == 0);
}

#define CHECK_RESULT(ret) CheckResult(__FILE__, __LINE__, ret)

int main(void) {
  assert(access("/etc/services", O_RDONLY) == 0);
  struct servent result_buf;
  struct servent *result;
  char buf[1024];
  // If these fail, check /etc/services if "ssh" exists. I picked this because
  // it should exist everywhere, if it doesn't, I am sorry. Disable the test
  // then please.
  CHECK_RESULT(
      getservbyname_r("ssh", nullptr, &result_buf, buf, sizeof(buf), &result));
  assert(result != nullptr);
  CHECK_RESULT(getservbyport_r(htons(22), nullptr, &result_buf, buf,
                               sizeof(buf), &result));
  assert(result != nullptr);

  CHECK_RESULT(getservent_r(&result_buf, buf, sizeof(buf), &result));
  assert(result != nullptr);

  CHECK_RESULT(getservbyname_r("invalidhadfuiasdhi", nullptr, &result_buf, buf,
                               sizeof(buf), &result));
  assert(result == nullptr);
}
