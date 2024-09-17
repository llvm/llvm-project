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
#include "src/sys/socket/recvmsg.h"
#include "src/sys/socket/sendmsg.h"
#include "src/sys/socket/socketpair.h"
#include "src/unistd/close.h"
#include "src/unistd/fork.h"

#include "src/errno/libc_errno.h"
#include "test/IntegrationTest/test.h"

#include <sys/socket.h> // For AF_UNIX and SOCK_STREAM

const char TEST_MESSAGE[] = "connection successful";
const size_t MESSAGE_LEN = sizeof(TEST_MESSAGE);

#define SEND_TEST_NAME "Sendmsg/Recvmsg"

TEST_MAIN(int argc, char **argv, char **envp) {

  const char *FILENAME = "send_file.test";
  // auto SOCK_PATH = libc_make_test_file_path(FILENAME);
  auto SOCK_PATH = FILENAME;

  // If the test fails, then the file for the socket may not be properly
  // removed. This ensures a consistent start.
  LIBC_NAMESPACE::remove(SOCK_PATH);
  LIBC_NAMESPACE::libc_errno = 0;

  int sockpair[2] = {0, 0};

  int result = LIBC_NAMESPACE::socketpair(AF_UNIX, SOCK_STREAM, 0, sockpair);
  ASSERT_EQ(result, 0);
  ASSERT_ERRNO_SUCCESS();

  LIBC_NAMESPACE::puts(SEND_TEST_NAME " Test Start");

  // split into client and server processes.
  pid_t pid = LIBC_NAMESPACE::fork();
  ASSERT_NE(pid, -1);
  ASSERT_ERRNO_SUCCESS();

  if (pid == 0) { // child
    LIBC_NAMESPACE::puts(SEND_TEST_NAME " Child Start");
    LIBC_NAMESPACE::close(sockpair[1]); // close parent sock

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

    ssize_t send_result = LIBC_NAMESPACE::sendmsg(sockpair[0], &message, 0);
    EXPECT_EQ(send_result, static_cast<ssize_t>(MESSAGE_LEN));
    ASSERT_ERRNO_SUCCESS();

    LIBC_NAMESPACE::close(sockpair[0]); // close child sock
    LIBC_NAMESPACE::puts(SEND_TEST_NAME " Child End");
  } else { // parent
    LIBC_NAMESPACE::puts(SEND_TEST_NAME " Parent Start");
    LIBC_NAMESPACE::close(sockpair[0]); // close child sock

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

    ssize_t recv_result = LIBC_NAMESPACE::recvmsg(sockpair[1], &message, 0);
    ASSERT_EQ(recv_result, MESSAGE_LEN);
    ASSERT_ERRNO_SUCCESS();

    for (size_t j = 0; buffer[j] != '\0' && TEST_MESSAGE[j] != '\0'; ++j) {
      ASSERT_EQ(buffer[j], TEST_MESSAGE[j]);
    }

    LIBC_NAMESPACE::close(sockpair[1]); // close parent sock
    LIBC_NAMESPACE::puts(SEND_TEST_NAME " Parent End");
  }
  return 0;
}
