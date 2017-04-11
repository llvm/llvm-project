// RUN: %clangxx_msan %s -DSEND -DPOISON -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=SEND
// RUN: %clangxx_msan %s -DSENDTO -DPOISON -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=SENDTO
// RUN: %clangxx_msan %s -DSENDMSG -DPOISON -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=SENDMSG

// RUN: %clangxx_msan %s -DSEND -o %t && %run %t 2>&1 | FileCheck %s --check-prefix=NEGATIVE
// RUN: %clangxx_msan %s -DSENDTO -o %t && %run %t 2>&1 | FileCheck %s --check-prefix=NEGATIVE
// RUN: %clangxx_msan %s -DSENDMSG -o %t && %run %t 2>&1 | FileCheck %s --check-prefix=NEGATIVE

// RUN: %clangxx_msan %s -DSEND -DPOISON -o %t && \
// RUN:   MSAN_OPTIONS=intercept_send=0 %run %t 2>&1 | FileCheck %s --check-prefix=NEGATIVE
// RUN: %clangxx_msan %s -DSENDTO -DPOISON -o %t && \
// RUN:   MSAN_OPTIONS=intercept_send=0 %run %t 2>&1 | FileCheck %s --check-prefix=NEGATIVE
// RUN: %clangxx_msan %s -DSENDMSG -DPOISON -o %t && \
// RUN:   MSAN_OPTIONS=intercept_send=0 %run %t 2>&1 | FileCheck %s --check-prefix=NEGATIVE

// UNSUPPORTED: android

#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sanitizer/msan_interface.h>

const int kBufSize = 10;
int sockfd;

int main() {
  int ret;
  char buf[kBufSize] = {0};
  pthread_t client_thread;
  struct sockaddr_in serveraddr;
  struct sockaddr_in6 serveraddr6;

  memset(&serveraddr, 0, sizeof(serveraddr));
  serveraddr.sin_family = AF_INET;
  serveraddr.sin_addr.s_addr = htonl(INADDR_ANY);
  serveraddr.sin_port = 0;
  struct sockaddr *addr = (struct sockaddr *)&serveraddr;
  socklen_t addrlen = sizeof(serveraddr);

  sockfd = socket(addr->sa_family, SOCK_DGRAM, 0);
  if (sockfd <= 0) {
    // Try to fall-back to IPv6
    memset(&serveraddr6, 0, sizeof(serveraddr6));
    serveraddr6.sin6_family = AF_INET6;
    serveraddr6.sin6_addr = in6addr_any;
    serveraddr6.sin6_port = 0;
    addr = (struct sockaddr *)&serveraddr6;
    addrlen = sizeof(serveraddr6);

    sockfd = socket(addr->sa_family, SOCK_DGRAM, 0);
  }
  assert(sockfd > 0);

  bind(sockfd, addr, addrlen);
  getsockname(sockfd, addr, &addrlen);

#if defined(POISON)
  __msan_poison(buf + 7, 1);
#endif

#if defined(SENDMSG)
  struct iovec iov[2] = {{buf, 5}, {buf + 5, 5}};
  struct msghdr msg;
  msg.msg_name = addr;
  msg.msg_namelen = addrlen;
  msg.msg_iov = iov;
  msg.msg_iovlen = 2;
  msg.msg_control = 0;
  msg.msg_controllen = 0;
  msg.msg_flags = 0;
#endif

#if defined(SEND)
  ret = connect(sockfd, addr, addrlen);
  assert(ret == 0);
  ret = send(sockfd, buf, kBufSize, 0);
  // SEND: Uninitialized bytes in __interceptor_send at offset 7 inside [{{.*}}, 10)
  assert(ret > 0);
#elif defined(SENDTO)
  ret = sendto(sockfd, buf, kBufSize, 0, addr, addrlen);
  // SENDTO: Uninitialized bytes in __interceptor_sendto at offset 7 inside [{{.*}}, 10)
  assert(ret > 0);
#elif defined(SENDMSG)
  ret = sendmsg(sockfd, &msg, 0);
  // SENDMSG: Uninitialized bytes in {{.*}} at offset 2 inside [{{.*}}, 5)
  assert(ret > 0);
#endif
  fprintf(stderr, "== done\n");
  // NEGATIVE: == done
  return 0;
}
