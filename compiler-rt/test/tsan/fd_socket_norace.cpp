// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include <arpa/inet.h>
#include <assert.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

struct sockaddr_in addr4;
struct sockaddr_in6 addr6;
struct sockaddr *addr;
socklen_t addrlen;
int X;

void *ClientThread(void *x) {
  X = 42;
  int c = socket(addr->sa_family, SOCK_STREAM, IPPROTO_TCP);
  if (connect(c, addr, addrlen)) {
    perror("connect");
    exit(1);
  }
  if (send(c, "a", 1, 0) != 1) {
    perror("send");
    exit(1);
  }
  close(c);
  return NULL;
}

int main() {
  addr4.sin_family = AF_INET;
  addr4.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr4.sin_port = INADDR_ANY;
  addr = (struct sockaddr *)&addr4;
  addrlen = sizeof(addr4);

  int s = socket(addr->sa_family, SOCK_STREAM, IPPROTO_TCP);
  if (s <= 0) {
    // Try to fall-back to IPv6
    addr6.sin6_family = AF_INET6;
    addr6.sin6_addr = in6addr_loopback;
    addr6.sin6_port = INADDR_ANY;
    addr = (struct sockaddr *)&addr6;
    addrlen = sizeof(addr6);
    s = socket(addr->sa_family, SOCK_STREAM, IPPROTO_TCP);
  }
  assert(s > 0);

  bind(s, addr, addrlen);
  getsockname(s, addr, &addrlen);
  listen(s, 10);
  pthread_t t;
  pthread_create(&t, 0, ClientThread, 0);
  int c = accept(s, 0, 0);
  char buf;
  while (read(c, &buf, 1) != 1) {
  }
  X = 43;
  close(c);
  close(s);
  pthread_join(t, 0);
  fprintf(stderr, "OK\n");
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race

