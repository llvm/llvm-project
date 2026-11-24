// RUN: %clangxx_msan -O0 %s -o %t && %run %t
// RUN: %clangxx_msan -O3 %s -o %t && %run %t

#include <assert.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>

int main(void) {
  struct addrinfo hints;

  hints.ai_flags = 0;
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = 0;
  hints.ai_addrlen = 0;
  hints.ai_addr = nullptr;
  hints.ai_canonname = nullptr;
  hints.ai_next = nullptr;

  struct addrinfo *res = nullptr;
  int ret = getaddrinfo("127.0.0.1", "4567", &hints, &res);
  assert(ret == 0);

  freeaddrinfo(res);
  return 0;
}
