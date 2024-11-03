// RUN: %clangxx_msan %s -DSEND -DPOISON -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=SEND
// RUN: %clangxx_msan %s -DSENDTO -DPOISON -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=SENDTO
// RUN: %clangxx_msan %s -DSENDMSG -DPOISON -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=SENDMSG
// RUN: %clangxx_msan %s -DSENDMMSG -DPOISON -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=SENDMMSG

// RUN: %clangxx_msan %s -DSEND -o %t && %run %t 2>&1 | FileCheck %s --check-prefix=NEGATIVE
// RUN: %clangxx_msan %s -DSENDTO -o %t && %run %t 2>&1 | FileCheck %s --check-prefix=NEGATIVE
// RUN: %clangxx_msan %s -DSENDMSG -o %t && %run %t 2>&1 | FileCheck %s --check-prefix=NEGATIVE
// RUN: %clangxx_msan %s -DSENDMMSG -o %t && %run %t 2>&1 | FileCheck %s --check-prefix=NEGATIVE

// RUN: %clangxx_msan %s -DSEND -DPOISON -o %t && \
// RUN:   MSAN_OPTIONS=intercept_send=0 %run %t 2>&1 | FileCheck %s --check-prefix=NEGATIVE
// RUN: %clangxx_msan %s -DSENDTO -DPOISON -o %t && \
// RUN:   MSAN_OPTIONS=intercept_send=0 %run %t 2>&1 | FileCheck %s --check-prefix=NEGATIVE
// RUN: %clangxx_msan %s -DSENDMSG -DPOISON -o %t && \
// RUN:   MSAN_OPTIONS=intercept_send=0 %run %t 2>&1 | FileCheck %s --check-prefix=NEGATIVE
// RUN: %clangxx_msan %s -DSENDMMSG -DPOISON -o %t && \
// RUN:   MSAN_OPTIONS=intercept_send=0 %run %t 2>&1 | FileCheck %s --check-prefix=NEGATIVE

// UNSUPPORTED: android

#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sanitizer/msan_interface.h>

const int kBufSize = 10;
const int kRecvBufSize = 100;
int sockfd[2];

int main() {
  int ret;
  int sent;
  char buf[kBufSize] = {0};
  char rbuf[kRecvBufSize];

  ret = socketpair(AF_LOCAL, SOCK_DGRAM, 0, sockfd);
  assert(!ret);

#if defined(POISON)
  __msan_poison(buf + 7, 1);
#endif

#if defined(SENDMSG) || defined(SENDMMSG)
  struct iovec iov[2] = {{buf, 5}, {buf + 5, 5}};
  struct msghdr msg;
  msg.msg_name = nullptr;
  msg.msg_namelen = 0;
  msg.msg_iov = iov;
  msg.msg_iovlen = 2;
  msg.msg_control = 0;
  msg.msg_controllen = 0;
  msg.msg_flags = 0;
#endif

#if defined(SENDMMSG)
  struct iovec iov0[1] = {{buf, 7}};
  struct msghdr msg0;
  msg0.msg_name = nullptr;
  msg0.msg_namelen = 0;
  msg0.msg_iov = iov0;
  msg0.msg_iovlen = 1;
  msg0.msg_control = 0;
  msg0.msg_controllen = 0;
  msg0.msg_flags = 0;

  struct mmsghdr mmsg[2];
  mmsg[0].msg_hdr = msg0; // good
  mmsg[1].msg_hdr = msg; // poisoned
#endif

#if defined(SEND)
  sent = send(sockfd[0], buf, kBufSize, 0);
  // SEND: Uninitialized bytes in send at offset 7 inside [{{.*}}, 10)
  assert(sent > 0);

  ret = recv(sockfd[1], rbuf, kRecvBufSize, 0);
  assert(ret == sent);
  assert(__msan_test_shadow(rbuf, kRecvBufSize) == sent);
#elif defined(SENDTO)
  sent = sendto(sockfd[0], buf, kBufSize, 0, nullptr, 0);
  // SENDTO: Uninitialized bytes in sendto at offset 7 inside [{{.*}}, 10)
  assert(sent > 0);

  struct sockaddr_storage ss;
  socklen_t sslen = sizeof(ss);
  ret = recvfrom(sockfd[1], rbuf, kRecvBufSize, 0, (struct sockaddr *)&ss,
                 &sslen);
  assert(ret == sent);
  assert(__msan_test_shadow(rbuf, kRecvBufSize) == sent);
  assert(__msan_test_shadow(&ss, sizeof(ss)) == sslen);
#elif defined(SENDMSG)
  sent = sendmsg(sockfd[0], &msg, 0);
  // SENDMSG: Uninitialized bytes in {{.*}} at offset 2 inside [{{.*}}, 5)
  assert(sent > 0);

  struct iovec riov[2] = {{rbuf, 3}, {rbuf + 3, kRecvBufSize - 3}};
  struct msghdr rmsg;
  rmsg.msg_name = nullptr;
  rmsg.msg_namelen = 0;
  rmsg.msg_iov = riov;
  rmsg.msg_iovlen = 2;
  rmsg.msg_control = 0;
  rmsg.msg_controllen = 0;
  rmsg.msg_flags = 0;

  ret = recvmsg(sockfd[1], &rmsg, 0);
  assert(ret == sent);
  assert(__msan_test_shadow(rbuf, kRecvBufSize) == sent);
#elif defined(SENDMMSG)
  sent = sendmmsg(sockfd[0], mmsg, 2, 0);
  // SENDMMSG: Uninitialized bytes in {{.*}} at offset 2 inside [{{.*}}, 5)
  assert(sent == 2);
  if (ret >= 2)
    assert(mmsg[1].msg_len > 0);

  struct iovec riov[2] = {{rbuf + kRecvBufSize / 2, kRecvBufSize / 2}};
  struct msghdr rmsg;
  rmsg.msg_name = nullptr;
  rmsg.msg_namelen = 0;
  rmsg.msg_iov = riov;
  rmsg.msg_iovlen = 1;
  rmsg.msg_control = 0;
  rmsg.msg_controllen = 0;
  rmsg.msg_flags = 0;

  struct iovec riov0[2] = {{rbuf, kRecvBufSize / 2}};
  struct msghdr rmsg0;
  rmsg0.msg_name = nullptr;
  rmsg0.msg_namelen = 0;
  rmsg0.msg_iov = riov0;
  rmsg0.msg_iovlen = 1;
  rmsg0.msg_control = 0;
  rmsg0.msg_controllen = 0;
  rmsg0.msg_flags = 0;

  struct mmsghdr rmmsg[2];
  rmmsg[0].msg_hdr = rmsg0;
  rmmsg[1].msg_hdr = rmsg;

  ret = recvmmsg(sockfd[1], rmmsg, 2, 0, nullptr);
  assert(ret == sent);
  assert(__msan_test_shadow(rbuf, kRecvBufSize) == 7);
  assert(__msan_test_shadow(rbuf + kRecvBufSize / 2, kRecvBufSize / 2) == 10);
#endif
  fprintf(stderr, "== done\n");
  // NEGATIVE: == done
  return 0;
}
