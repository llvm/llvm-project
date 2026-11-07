//===-- Unittests for the netinet/tcp.h macros ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for the netinet/tcp.h macros.
///
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/netinet-tcp-macros.h"
#include "test/UnitTest/Test.h"

// The kernel names the control message type of TCP_INQ separately, and
// callers read the byte count back by that name.
TEST(LlvmLibcNetinetTcpTest, InqControlMessageHasTheOptionValue) {
  EXPECT_EQ(TCP_CM_INQ, TCP_INQ);
}
