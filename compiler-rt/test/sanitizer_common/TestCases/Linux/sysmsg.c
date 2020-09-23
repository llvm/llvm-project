// RUN: %clang -O1 %s -o %t && %run %t
// UNSUPPORTED: android
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/msg.h>

#define CHECK_STRING "hello, world!"
#define MSG_BUFLEN 0x100

int main() {
  int msgq = msgget(IPC_PRIVATE, 0666);
  if (msgq == -1) perror("msgget:");
  assert(msgq != -1);

  struct msg_s {
    long mtype;
    char string[MSG_BUFLEN];
  };

  struct msg_s msg = {
      .mtype = 1};
  strcpy(msg.string, CHECK_STRING);
  int res = msgsnd(msgq, &msg, MSG_BUFLEN, IPC_NOWAIT);
  if (res) {
    fprintf(stderr, "Error sending message! %s\n", strerror(errno));
    return -1;
  }

  struct msg_s rcv_msg;
  ssize_t len = msgrcv(msgq, &rcv_msg, MSG_BUFLEN, msg.mtype, IPC_NOWAIT);
  assert(len == MSG_BUFLEN);
  assert(msg.mtype == rcv_msg.mtype);
  assert(!memcmp(msg.string, rcv_msg.string, MSG_BUFLEN));
  msgctl(msgq, IPC_RMID, NULL);
  return 0;
}
