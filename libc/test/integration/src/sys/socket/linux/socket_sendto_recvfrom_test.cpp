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
#include "src/sys/socket/recvfrom.h"
#include "src/sys/socket/sendto.h"
#include "src/sys/socket/socketpair.h"
#include "src/unistd/close.h"
#include "src/unistd/fork.h"

#include "src/errno/libc_errno.h"
#include "test/IntegrationTest/test.h"

#include <sys/socket.h> // For AF_UNIX and SOCK_STREAM

const char TEST_MESSAGE[] = "connection successful";
const size_t MESSAGE_LEN = sizeof(TEST_MESSAGE);

// macro for easy string pasting, we don't need printf here
#define SEND_TEST_NAME "Sendto/Recvfrom"

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

    ssize_t send_result = LIBC_NAMESPACE::sendto(sockpair[0], TEST_MESSAGE,
                                                 MESSAGE_LEN, 0, nullptr, 0);
    EXPECT_EQ(send_result, static_cast<ssize_t>(MESSAGE_LEN));
    ASSERT_ERRNO_SUCCESS();

    LIBC_NAMESPACE::close(sockpair[0]); // close child sock
    LIBC_NAMESPACE::puts(SEND_TEST_NAME " Child End");
  } else { // parent
    LIBC_NAMESPACE::puts(SEND_TEST_NAME " Parent Start");
    LIBC_NAMESPACE::close(sockpair[0]); // close child sock

    char buffer[256];

    ssize_t recv_result = LIBC_NAMESPACE::recvfrom(
        sockpair[1], buffer, sizeof(buffer), 0, nullptr, 0);
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
