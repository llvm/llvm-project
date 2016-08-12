// RUN: %clangxx_msan %s -std=c++11 -DSENDMSG -DPOISONFD -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=SENDMSG
// RUN: %clangxx_msan %s -std=c++11 -DSENDMSG -DPOISONCRED -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=SENDMSG
// RUN: %clangxx_msan %s -std=c++11 -DSENDMSG -DPOISONLEN -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=SENDMSG
// RUN: %clangxx_msan %s -std=c++11 -DSENDMSG -DPOISONLEVEL -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=SENDMSG
// RUN: %clangxx_msan %s -std=c++11 -DSENDMSG -DPOISONTYPE -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=SENDMSG
// RUN: %clangxx_msan %s -std=c++11 -DSENDMSG -DPOISONLEN2 -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=SENDMSG
// RUN: %clangxx_msan %s -std=c++11 -DSENDMSG -DPOISONLEVEL2 -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=SENDMSG
// RUN: %clangxx_msan %s -std=c++11 -DSENDMSG -DPOISONTYPE2 -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=SENDMSG
// RUN: %clangxx_msan %s -std=c++11 -DSENDMSG -o %t && %run %t 2>&1 | FileCheck %s --check-prefix=NEGATIVE

// UNSUPPORTED: android

// XFAIL: target-is-mips64el

#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sanitizer/msan_interface.h>

const int kBufSize = 10;

int main() {
  int ret;
  char buf[kBufSize] = {0};
  pthread_t client_thread;
  struct sockaddr_un serveraddr;

  int sock[2];
  ret = socketpair(AF_UNIX, SOCK_STREAM, 0, sock);
  assert(ret == 0);

  int sockfd = sock[0];

  struct iovec iov[] = {{buf, 10}};
  struct msghdr msg = {0};
  msg.msg_iov = iov;
  msg.msg_iovlen = 1;
  msg.msg_flags = 0;

  static const int kNumFds = 3;
  char controlbuf[CMSG_SPACE(kNumFds * sizeof(int)) +
                  CMSG_SPACE(sizeof(struct ucred))];
  msg.msg_control = &controlbuf;
  msg.msg_controllen = sizeof(controlbuf);

  struct cmsghdr *cmsg = (struct cmsghdr *)&controlbuf;
  assert(cmsg);
  int myfds[kNumFds];
  for (int &fd : myfds)
    fd = sockfd;
#ifdef POISONFD
  __msan_poison(&myfds[1], sizeof(int));
#endif
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(kNumFds * sizeof(int));
  memcpy(CMSG_DATA(cmsg), myfds, kNumFds * sizeof(int));
#ifdef POISONLEVEL
  __msan_poison(&cmsg->cmsg_level, sizeof(cmsg->cmsg_level));
#endif
#ifdef POISONTYPE
  __msan_poison(&cmsg->cmsg_type, sizeof(cmsg->cmsg_type));
#endif
#ifdef POISONLEN
  __msan_poison(&cmsg->cmsg_len, sizeof(cmsg->cmsg_len));
#endif

  cmsg = (struct cmsghdr *)(&controlbuf[CMSG_SPACE(kNumFds * sizeof(int))]);
  assert(cmsg);
  struct ucred cred = {getpid(), getuid(), getgid()};
#ifdef POISONCRED
  __msan_poison(&cred.uid, sizeof(cred.uid));
#endif
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_CREDENTIALS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(struct ucred));
  memcpy(CMSG_DATA(cmsg), &cred, sizeof(struct ucred));
#ifdef POISONLEVEL2
  __msan_poison(&cmsg->cmsg_level, sizeof(cmsg->cmsg_level));
#endif
#ifdef POISONTYPE2
  __msan_poison(&cmsg->cmsg_type, sizeof(cmsg->cmsg_type));
#endif
#ifdef POISONLEN2
  __msan_poison(&cmsg->cmsg_len, sizeof(cmsg->cmsg_len));
#endif

  ret = sendmsg(sockfd, &msg, 0);
  // SENDMSG: MemorySanitizer: use-of-uninitialized-value
  if (ret == -1) printf("%d: %s\n", errno, strerror(errno));
  assert(ret > 0);

  fprintf(stderr, "== done\n");
  // NEGATIVE: == done
  return 0;
}
