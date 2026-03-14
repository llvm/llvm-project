// Test that ASan detects buffer overflow on read from socket via recvfrom.
//
// RUN: %clangxx_asan %s -DRECVFROM -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-RECVFROM
// RUN: %clangxx_asan %s -DSENDTO -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-SENDTO
// RUN: %clangxx_asan %s -DSENDTO -o %t && %env_asan_opts=intercept_send=0 %run %t 2>&1
//
// This will try to fast unwind on Arm Thumb, where fast unwinding does not work.
// UNSUPPORTED: android, !fast-unwinder-works

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <pthread.h>

#define CHECK_ERROR(p, m)                                                      \
  do {                                                                         \
    if (p) {                                                                   \
      fprintf(stderr, "ERROR " m "\n");                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

const int kBufSize = 10;
int sockfd;

static void *client_thread_udp(void *data) {
#ifdef SENDTO
  const char buf[kBufSize / 2] = {0, };
#else
  const char buf[kBufSize] = {0, };
#endif
  struct sockaddr_in serveraddr;
  socklen_t addrlen = sizeof(serveraddr);

  int succeeded = getsockname(sockfd, (struct sockaddr *)&serveraddr, &addrlen);
  CHECK_ERROR(succeeded < 0, "in getsockname");

  succeeded = sendto(sockfd, buf, kBufSize, 0, (struct sockaddr *)&serveraddr,
                     sizeof(serveraddr));
  // CHECK-SENDTO: {{READ of size 10 at 0x.* thread T1}}
  // CHECK-SENDTO: {{    #1 0x.* in client_thread_udp.*recvfrom.cpp:}}[[@LINE-3]]
  CHECK_ERROR(succeeded < 0, "in sending message");
  return NULL;
}

int main() {
#ifdef RECVFROM
  char buf[kBufSize / 2];
#else
  char buf[kBufSize];
#endif
  pthread_t client_thread;
  struct sockaddr_in serveraddr;

  sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  CHECK_ERROR(sockfd < 0, "opening socket");

  memset(&serveraddr, 0, sizeof(serveraddr));
  serveraddr.sin_family = AF_INET;
  serveraddr.sin_addr.s_addr = htonl(INADDR_ANY);
  serveraddr.sin_port = 0;

  int bound = bind(sockfd, (struct sockaddr *)&serveraddr, sizeof(serveraddr));
  CHECK_ERROR(bound < 0, "on binding");

  int succeeded =
      pthread_create(&client_thread, NULL, client_thread_udp, &serveraddr);
  CHECK_ERROR(succeeded, "creating thread");

  recvfrom(sockfd, buf, kBufSize, 0, NULL, NULL); // BOOM
  // CHECK-RECVFROM: {{WRITE of size 10 at 0x.* thread T0}}
  // CHECK-RECVFROM: {{    #1 0x.* in main.*recvfrom.cpp:}}[[@LINE-2]]
  // CHECK-RECVFROM: {{Address 0x.* is located in stack of thread T0 at offset}}
  // CHECK-RECVFROM-NEXT: in{{.*}}main{{.*}}recvfrom.cpp
  succeeded = pthread_join(client_thread, NULL);
  CHECK_ERROR(succeeded, "joining thread");
  return 0;
}
