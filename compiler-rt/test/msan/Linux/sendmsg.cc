// RUN: %clangxx_msan %s -DSEND -DBUF -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=SEND
// RUN: %clangxx_msan %s -DSENDTO -DBUF -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=SENDTO
// RUN: %clangxx_msan %s -DSENDMSG -DBUF -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=SENDMSG

// FIXME: intercept connect() and add a SEND+ADDR test
// RUN: %clangxx_msan %s -DSENDTO -DADDR -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=SENDTO-ADDR
// RUN: %clangxx_msan %s -DSENDMSG -DADDR -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=SENDMSG-ADDR

// RUN: %clangxx_msan %s -DSEND -o %t && %run %t 2>&1 | FileCheck %s --check-prefix=NEGATIVE
// RUN: %clangxx_msan %s -DSENDTO -o %t && %run %t 2>&1 | FileCheck %s --check-prefix=NEGATIVE
// RUN: %clangxx_msan %s -DSENDMSG -o %t && %run %t 2>&1 | FileCheck %s --check-prefix=NEGATIVE

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

  sockfd = socket(AF_INET, SOCK_DGRAM, 0);

  memset(&serveraddr, 0, sizeof(serveraddr));
  serveraddr.sin_family = AF_INET;
  serveraddr.sin_addr.s_addr = htonl(INADDR_ANY);
  serveraddr.sin_port = 0;

  bind(sockfd, (struct sockaddr *)&serveraddr, sizeof(serveraddr));
  socklen_t addrlen = sizeof(serveraddr);
  getsockname(sockfd, (struct sockaddr *)&serveraddr, &addrlen);

#if defined(ADDR)
  assert(addrlen > 3);
  __msan_poison(((char *)&serveraddr) + 3, 1);
#elif defined(BUF)
  __msan_poison(buf + 7, 1);
#endif

#if defined(SENDMSG)
  struct iovec iov[2] = {{buf, 5}, {buf + 5, 5}};
  struct msghdr msg;
  memset(&msg, 0, sizeof(msg));
  msg.msg_name = &serveraddr;
  msg.msg_namelen = addrlen;
  msg.msg_iov = iov;
  msg.msg_iovlen = 2;
#endif

#if defined(SEND)
  ret = connect(sockfd, (struct sockaddr *)&serveraddr, addrlen);
  assert(ret == 0);
  ret = send(sockfd, buf, kBufSize, 0);
  // SEND: Uninitialized bytes in __interceptor_send at offset 7 inside [{{.*}}, 10)
  assert(ret > 0);
#elif defined(SENDTO)
  ret =
      sendto(sockfd, buf, kBufSize, 0, (struct sockaddr *)&serveraddr, addrlen);
  // SENDTO: Uninitialized bytes in __interceptor_sendto at offset 7 inside [{{.*}}, 10)
  // SENDTO-ADDR: Uninitialized bytes in __interceptor_sendto at offset 3 inside [{{.*}},
  assert(ret > 0);
#elif defined(SENDMSG)
  ret = sendmsg(sockfd, &msg, 0);
  // SENDMSG: Uninitialized bytes in {{.*}} at offset 2 inside [{{.*}}, 5)
  // SENDMSG-ADDR: Uninitialized bytes in {{.*}} at offset 3 inside [{{.*}},
  assert(ret > 0);
#endif
  fprintf(stderr, "== done\n");
  // NEGATIVE: == done
  return 0;
}
