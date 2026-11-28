//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for netinet/tcp.h.
///
//===----------------------------------------------------------------------===//

#include "hdr/types/struct_tcp_info.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcNetinetTcpTest, StructTcpInfoLayout) {
  // These fields have been around forever, so they should be present in any
  // version of the struct.
  EXPECT_EQ(offsetof(struct tcp_info, tcpi_state), static_cast<size_t>(0));
  EXPECT_EQ(offsetof(struct tcp_info, tcpi_options), static_cast<size_t>(5));
  EXPECT_EQ(offsetof(struct tcp_info, tcpi_rto), static_cast<size_t>(8));
  EXPECT_EQ(offsetof(struct tcp_info, tcpi_total_retrans),
            static_cast<size_t>(100));

#ifdef LIBC_FULL_BUILD
  // In overlay mode, the system headers may contain an older version of the
  // struct, so only check our own definition.
  EXPECT_EQ(offsetof(struct tcp_info, tcpi_pacing_rate),
            static_cast<size_t>(104));
  EXPECT_EQ(offsetof(struct tcp_info, tcpi_delivery_rate),
            static_cast<size_t>(160));
  EXPECT_EQ(offsetof(struct tcp_info, tcpi_bytes_sent),
            static_cast<size_t>(200));
  EXPECT_EQ(offsetof(struct tcp_info, tcpi_total_rto),
            static_cast<size_t>(240));
  EXPECT_EQ(offsetof(struct tcp_info, tcpi_received_ce),
            static_cast<size_t>(248));
  EXPECT_EQ(offsetof(struct tcp_info, tcpi_received_ce_bytes),
            static_cast<size_t>(272));
  EXPECT_EQ(sizeof(struct tcp_info), static_cast<size_t>(280));
#endif
}
