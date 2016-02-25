// Test that ASan detects buffer overflow on read from socket via recvfrom.
//
// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <pthread.h>

const int kPortNum = 1234;
const int kBufSize = 10;

static void *server_thread_udp(void *data) {
  char buf[kBufSize / 2];
  struct sockaddr_in serveraddr; // server's addr
  int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  if (sockfd < 0)
    fprintf(stderr, "ERROR opening socket\n");

  memset((char *) &serveraddr, 0, sizeof(serveraddr));
  serveraddr.sin_family = AF_INET;
  serveraddr.sin_addr.s_addr = htonl(INADDR_ANY);
  serveraddr.sin_port = htons(kPortNum);

  if (bind(sockfd, (struct sockaddr *) &serveraddr, sizeof(serveraddr)) < 0)
    fprintf(stderr, "ERROR on binding\n");

  recvfrom(sockfd, buf, kBufSize, 0, NULL, NULL); // BOOM
  // CHECK: {{WRITE of size 10 at 0x.* thread T1}}
  // CHECK: {{    #1 0x.* in server_thread_udp.*recvfrom.cc:}}[[@LINE-2]]
  // CHECK: {{Address 0x.* is located in stack of thread T1 at offset}}
  // CHECK-NEXT: in{{.*}}server_thread_udp{{.*}}recvfrom.cc
  return NULL;
}

int main() {
  char buf[kBufSize] = "123456789";
  struct sockaddr_in serveraddr; // server's addr

  pthread_t server_thread;
  if (pthread_create(&server_thread, NULL, server_thread_udp, NULL)) {
    fprintf(stderr, "Error creating thread\n");
    exit(1);
  }

  int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  struct hostent *server;
  char hostname[] = "localhost";
  if (sockfd < 0)
    fprintf(stderr, "ERROR opening socket\n");

  server = gethostbyname(hostname);
  if (!server) {
    fprintf(stderr,"ERROR, no such host as %s\n", hostname);
    exit(1);
  }

  memset((char *) &serveraddr, 0, sizeof(serveraddr));
  serveraddr.sin_family = AF_INET;
  memcpy((char *)&serveraddr.sin_addr.s_addr, (char *)server->h_addr,
          server->h_length);
  serveraddr.sin_port = htons(kPortNum);
  sendto(sockfd, buf, strlen(buf), 0, (struct sockaddr *) &serveraddr,
         sizeof(serveraddr));

  if (pthread_join(server_thread, NULL)) {
    fprintf(stderr, "Error joining thread\n");
    exit(1);
  }

  return 0;
}
