// Test that ASan doesn't raise false alarm when MSG_TRUNC is present.
//
// RUN: %clangxx %s -o %t && %run %t 2>&1
//
// UNSUPPORTED: android

#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/ip.h>
#include <assert.h>

int main() {
  int fd_0 = socket(AF_INET, SOCK_DGRAM, 0);
  int fd_1 = socket(AF_INET, SOCK_DGRAM, 0);
  struct sockaddr_in sin;
  socklen_t len = sizeof(sin);
  char *buf = (char *)malloc(1);

  sin.sin_family = AF_INET;
  // Choose a random port to bind.
  sin.sin_port = 0;
  sin.sin_addr.s_addr = INADDR_ANY;

  assert(bind(fd_1, (struct sockaddr *)&sin, sizeof(sin)) == 0);
  // Get the address and port binded.
  assert(getsockname(fd_1, (struct sockaddr *)&sin, &len) == 0);
  assert(sendto(fd_0, "hello", strlen("hello"), MSG_DONTWAIT,
                (struct sockaddr *)&sin, sizeof(sin)) != -1);
  assert(recv(fd_1, buf, 1, MSG_TRUNC) != -1);
  free(buf);

  return 0;
}

