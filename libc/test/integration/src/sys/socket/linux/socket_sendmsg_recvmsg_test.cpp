//===-- Integration tests for socket functions ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/threads/sleep.h"

#include "src/stdio/puts.h"
#include "src/stdio/remove.h"
#include "src/sys/socket/accept.h"
#include "src/sys/socket/bind.h"
#include "src/sys/socket/connect.h"
#include "src/sys/socket/listen.h"
#include "src/sys/socket/recvmsg.h"
#include "src/sys/socket/sendmsg.h"
#include "src/sys/socket/socket.h"
#include "src/unistd/close.h"
#include "src/unistd/fork.h"

#include "src/errno/libc_errno.h"
#include "test/IntegrationTest/test.h"

#include <sys/socket.h> // For AF_UNIX and SOCK_STREAM

const char TEST_MESSAGE[] = "connection successful";
const size_t MESSAGE_LEN = sizeof(TEST_MESSAGE);

void run_client(const char *SOCK_PATH) {
  constexpr size_t SPIN_TRIES = 1000;
  int result;

  int sock = LIBC_NAMESPACE::socket(AF_UNIX, SOCK_STREAM, 0);
  ASSERT_NE(sock, -1);
  ASSERT_ERRNO_SUCCESS();

  struct sockaddr_un my_addr;

  my_addr.sun_family = AF_UNIX;
  size_t i = 0;
  for (; SOCK_PATH[i] != '\0' && (i < sizeof(my_addr.sun_path)); ++i)
    my_addr.sun_path[i] = SOCK_PATH[i];
  my_addr.sun_path[i] = '\0';

  // If the connection fails with "EAGAIN" then sleep breifly and try again, but
  // only up to SPIN_TRIES times.
  LIBC_NAMESPACE::libc_errno = EAGAIN;
  for (size_t j = 0; j < SPIN_TRIES && LIBC_NAMESPACE::libc_errno == EAGAIN;
       ++j, LIBC_NAMESPACE::sleep_briefly()) {
    LIBC_NAMESPACE::libc_errno = 0;
    result = LIBC_NAMESPACE::connect(
        sock, reinterpret_cast<struct sockaddr *>(&my_addr),
        sizeof(my_addr.sun_path));
  }
  ASSERT_EQ(result, 0);
  ASSERT_ERRNO_SUCCESS();

  iovec msg_text;
  msg_text.iov_base =
      reinterpret_cast<void *>(const_cast<char *>(TEST_MESSAGE));
  msg_text.iov_len = MESSAGE_LEN;

  msghdr message;
  message.msg_name = nullptr;
  message.msg_namelen = 0;
  message.msg_iov = &msg_text;
  message.msg_iovlen = 1;
  message.msg_control = nullptr;
  message.msg_controllen = 0;
  message.msg_flags = 0;

  ssize_t send_result = LIBC_NAMESPACE::sendmsg(sock, &message, 0);

  EXPECT_EQ(send_result, static_cast<ssize_t>(MESSAGE_LEN));
  ASSERT_ERRNO_SUCCESS();

  LIBC_NAMESPACE::close(sock);
}

void run_server(const char *SOCK_PATH) {
  int result;

  int sock = LIBC_NAMESPACE::socket(AF_UNIX, SOCK_STREAM, 0);
  ASSERT_NE(sock, -1);
  ASSERT_ERRNO_SUCCESS();

  struct sockaddr_un my_addr;

  my_addr.sun_family = AF_UNIX;
  size_t i = 0;
  for (; SOCK_PATH[i] != '\0' && (i < sizeof(my_addr.sun_path)); ++i)
    my_addr.sun_path[i] = SOCK_PATH[i];
  my_addr.sun_path[i] = '\0';

  result =
      LIBC_NAMESPACE::bind(sock, reinterpret_cast<struct sockaddr *>(&my_addr),
                           sizeof(my_addr.sun_path));
  ASSERT_EQ(result, 0);
  ASSERT_ERRNO_SUCCESS();

  result = LIBC_NAMESPACE::listen(sock, 1);
  ASSERT_EQ(result, 0);
  ASSERT_ERRNO_SUCCESS();

  struct sockaddr connected_sock;
  socklen_t sockaddr_len = sizeof(struct sockaddr);

  int accepted_sock =
      LIBC_NAMESPACE::accept(sock, &connected_sock, &sockaddr_len);
  ASSERT_NE(accepted_sock, -1);
  ASSERT_ERRNO_SUCCESS();

  char buffer[256];

  iovec msg_text;
  msg_text.iov_base = reinterpret_cast<void *>(buffer);
  msg_text.iov_len = sizeof(buffer);

  msghdr message;
  message.msg_name = nullptr;
  message.msg_namelen = 0;
  message.msg_iov = &msg_text;
  message.msg_iovlen = 1;
  message.msg_control = nullptr;
  message.msg_controllen = 0;
  message.msg_flags = 0;

  ssize_t recv_result = LIBC_NAMESPACE::recvmsg(accepted_sock, &message, 0);
  ASSERT_EQ(recv_result, MESSAGE_LEN);
  ASSERT_ERRNO_SUCCESS();

  for (size_t j = 0; buffer[j] != '\0' && TEST_MESSAGE[j] != '\0'; ++j) {
    ASSERT_EQ(buffer[j], TEST_MESSAGE[j]);
  }

  LIBC_NAMESPACE::close(sock);
}

TEST_MAIN(int argc, char **argv, char **envp) {

  const char *FILENAME = "sendmsg_file.test";
  // auto SOCK_PATH = libc_make_test_file_path(FILENAME);
  auto SOCK_PATH = FILENAME;

  // If the test fails, then the file for the socket may not be properly
  // removed. This ensures a consistent start.
  LIBC_NAMESPACE::remove(SOCK_PATH);
  LIBC_NAMESPACE::libc_errno = 0;

  LIBC_NAMESPACE::puts("Sendmsg/Recvmsg Test Start");

  // split into client and server processes.
  pid_t pid = LIBC_NAMESPACE::fork();
  ASSERT_NE(pid, -1);
  ASSERT_ERRNO_SUCCESS();

  if (pid == 0) { // child
    LIBC_NAMESPACE::puts("Sendmsg/Recvmsg Child Start");
    run_client(SOCK_PATH);
    LIBC_NAMESPACE::puts("Sendmsg/Recvmsg Child End");
  } else { // parent
    LIBC_NAMESPACE::puts("Sendmsg/Recvmsg Parent Start");
    run_server(SOCK_PATH);
    LIBC_NAMESPACE::puts("Sendmsg/Recvmsg Parent End");
  }
  return 0;
}
