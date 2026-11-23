// RUN: %clangxx -O0 -g %s -o %t && %run %t

#include <assert.h>
#include <netdb.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>

int main() {
  struct addrinfo hints;
  memset(&hints, 0, sizeof(hints));

  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_NUMERICHOST | AI_NUMERICSERV;

  struct addrinfo *res = nullptr;
  int ret = getaddrinfo("127.0.0.1", "4567", &hints, &res);

  assert(ret == 0);
  assert(res != nullptr);

  freeaddrinfo(res);
  return 0;
}
