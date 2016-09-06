// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

struct sockaddr_in addr;
int X;

void *ClientThread(void *x) {
  X = 42;
  int c = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (connect(c, (struct sockaddr*)&addr, sizeof(addr))) {
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
  int s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  addr.sin_family = AF_INET;
  inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
  addr.sin_port = INADDR_ANY;
  socklen_t len = sizeof(addr);
  bind(s, (sockaddr*)&addr, len);
  getsockname(s, (sockaddr*)&addr, &len);
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

