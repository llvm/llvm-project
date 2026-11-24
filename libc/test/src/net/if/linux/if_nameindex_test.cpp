//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for if_nameindex and if_freenameindex.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/net_if_macros.h"
#include "hdr/types/struct_if_nameindex.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/CPP/tuple.h"
#include "src/__support/error_or.h"
#include "src/__support/net/linux/netlink.h"
#include "src/net/if/if_freenameindex.h"
#include "src/net/if/if_nameindex.h"
#include "src/net/if/linux/if_nameindex_impl.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"
#include "test/src/__support/net/linux/netlink_test_support.h"

#include <linux/if_link.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>

using LIBC_NAMESPACE::Error;
using LIBC_NAMESPACE::cpp::get;
using LIBC_NAMESPACE::cpp::span;
using LIBC_NAMESPACE::net::MAX_DUMP_ATTEMPTS;
using LIBC_NAMESPACE::testing::AttrInt;
using LIBC_NAMESPACE::testing::AttrString;
using LIBC_NAMESPACE::testing::build_generic_nlmsg_packet;
using LIBC_NAMESPACE::testing::build_nlmsg_done_packet;
using LIBC_NAMESPACE::testing::build_nlmsg_error_packet;
using LIBC_NAMESPACE::testing::build_nlmsg_packet;
using LIBC_NAMESPACE::testing::FakeNetworkSyscallPolicy;
using LIBC_NAMESPACE::testing::FakeNetworkSyscallPolicyData;
using LIBC_NAMESPACE::testing::mark_interrupted;

namespace {

struct LlvmLibcIfNameIndexTest
    : public LIBC_NAMESPACE::testing::ErrnoCheckingTest {
  static constexpr int FAKE_SOCKET = 47;
  static FakeNetworkSyscallPolicyData policy_data;
  using Policy = FakeNetworkSyscallPolicy<&policy_data>;
  using Socket = LIBC_NAMESPACE::net::NetlinkSocket<Policy>;

  void SetUp() override {
    ErrnoCheckingTest::SetUp();
    policy_data = {};
  }
  void validate_dump_request(size_t num_requests = 1) {
    constexpr size_t REQUEST_LEN =
        static_cast<size_t>(NLMSG_LENGTH(sizeof(ifinfomsg)));
    ASSERT_EQ(policy_data.sendto_calls.size(), num_requests);
    ASSERT_EQ(policy_data.sendto_data.size(), num_requests * REQUEST_LEN);
    for (size_t i = 0; i < num_requests; ++i) {
      ASSERT_EQ(get<0>(policy_data.sendto_calls[i]), FAKE_SOCKET);
      ASSERT_EQ(get<2>(policy_data.sendto_calls[i]), 0);
      auto *nlh = reinterpret_cast<struct nlmsghdr *>(
          policy_data.sendto_data.data() + i * REQUEST_LEN);
      ASSERT_EQ(nlh->nlmsg_len, static_cast<uint32_t>(REQUEST_LEN));
      ASSERT_EQ(nlh->nlmsg_type, static_cast<uint16_t>(RTM_GETLINK));
      ASSERT_EQ(nlh->nlmsg_flags,
                static_cast<uint16_t>(NLM_F_REQUEST | NLM_F_DUMP));
      auto *ifm = reinterpret_cast<struct ifinfomsg *>(NLMSG_DATA(nlh));
      ASSERT_EQ(ifm->ifi_family, static_cast<unsigned char>(AF_UNSPEC));
    }
  }
};

struct LlvmLibcIfNameIndexSocketTest : public LlvmLibcIfNameIndexTest {
  void SetUp() override {
    LlvmLibcIfNameIndexTest::SetUp();
    policy_data.socket_results.push_back(FAKE_SOCKET);
  }

  void TearDown() override {
    ASSERT_EQ(policy_data.socket_calls.size(), size_t(1));
    ASSERT_EQ(get<0>(policy_data.socket_calls[0]), AF_NETLINK);
    ASSERT_EQ(get<1>(policy_data.socket_calls[0]), SOCK_RAW | SOCK_CLOEXEC);
    ASSERT_EQ(get<2>(policy_data.socket_calls[0]), NETLINK_ROUTE);

    ASSERT_EQ(policy_data.close_calls.size(), size_t(1));
    ASSERT_EQ(policy_data.close_calls[0], FAKE_SOCKET);
    ASSERT_TRUE(policy_data.sendto_results.empty());
    ASSERT_TRUE(policy_data.recv_results.empty());
    LlvmLibcIfNameIndexTest::TearDown();
  }
};

/// Name is the attribute we're most interested in. The others are not
/// important. We're just testing that we can skip over them correctly.
using AttrName = AttrString<IFLA_IFNAME>;
using AttrMtu = AttrInt<IFLA_MTU>;
using AttrTxqlen = AttrInt<IFLA_TXQLEN>;

} // namespace

FakeNetworkSyscallPolicyData LlvmLibcIfNameIndexTest::policy_data;

template <typename... Attrs>
static size_t build_ifinfomsg_packet(uint8_t *buf, unsigned int index,
                                     const Attrs &...attrs) {
  struct ifinfomsg ifm = {};
  ifm.ifi_family = AF_UNSPEC;
  ifm.ifi_index = static_cast<int>(index);
  return build_nlmsg_packet(buf, RTM_NEWLINK, ifm, attrs...);
}

TEST_F(LlvmLibcIfNameIndexTest, SocketCreationFailure) {
  policy_data.socket_results.push_back(Error(EAFNOSUPPORT));

  auto res = LIBC_NAMESPACE::net::if_nameindex<Policy>();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), EAFNOSUPPORT);
  ASSERT_EQ(policy_data.socket_calls.size(), size_t(1));
  ASSERT_EQ(get<0>(policy_data.socket_calls[0]), AF_NETLINK);
  ASSERT_EQ(get<1>(policy_data.socket_calls[0]), SOCK_RAW | SOCK_CLOEXEC);
  ASSERT_EQ(get<2>(policy_data.socket_calls[0]), NETLINK_ROUTE);
  ASSERT_EQ(policy_data.close_calls.size(), size_t(0));
}

TEST_F(LlvmLibcIfNameIndexSocketTest, SendFailure) {
  policy_data.sendto_results.push_back(Error(ENETDOWN));

  auto res = LIBC_NAMESPACE::net::if_nameindex<Policy>();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), ENETDOWN);
  validate_dump_request();
}

TEST_F(LlvmLibcIfNameIndexSocketTest, ZeroInterfaces) {
  uint8_t pkt_buf[128];
  size_t len = build_nlmsg_done_packet(pkt_buf);

  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf, len));

  auto res = LIBC_NAMESPACE::net::if_nameindex<Policy>();
  ASSERT_TRUE(res.has_value());
  struct if_nameindex *list = res.value();
  ASSERT_NE(list, static_cast<struct if_nameindex *>(nullptr));

  ASSERT_EQ(list[0].if_index, 0u);
  ASSERT_EQ(list[0].if_name, static_cast<char *>(nullptr));

  validate_dump_request();
  LIBC_NAMESPACE::if_freenameindex(list);
}

TEST_F(LlvmLibcIfNameIndexSocketTest, SingleInterface) {
  uint8_t pkt_buf[1024];
  size_t len1 = build_ifinfomsg_packet(pkt_buf, 1, AttrName{"lo"});
  size_t len2 = build_nlmsg_done_packet(pkt_buf + len1);

  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf, len1 + len2));

  auto res = LIBC_NAMESPACE::net::if_nameindex<Policy>();
  ASSERT_TRUE(res.has_value());
  struct if_nameindex *list = res.value();
  ASSERT_NE(list, static_cast<struct if_nameindex *>(nullptr));

  ASSERT_EQ(list[0].if_index, 1u);
  ASSERT_STREQ(list[0].if_name, "lo");
  ASSERT_EQ(list[1].if_index, 0u);
  ASSERT_EQ(list[1].if_name, static_cast<char *>(nullptr));

  validate_dump_request();

  ASSERT_EQ(policy_data.recv_calls.size(), size_t(1));
  ASSERT_EQ(get<0>(policy_data.recv_calls[0]), FAKE_SOCKET);

  LIBC_NAMESPACE::if_freenameindex(list);
}

TEST_F(LlvmLibcIfNameIndexSocketTest, MultipleInterfaces) {
  uint8_t pkt_buf[2048];
  size_t len1 = build_ifinfomsg_packet(pkt_buf, 1, AttrName{"lo"});
  size_t len2 = build_ifinfomsg_packet(pkt_buf + len1, 2, AttrName{"eth0"});
  size_t len3 =
      build_ifinfomsg_packet(pkt_buf + len1 + len2, 3, AttrName{"wlan0"});
  size_t len4 = build_nlmsg_done_packet(pkt_buf + len1 + len2 + len3);

  policy_data.recv_results.push_back(
      span<const uint8_t>(pkt_buf, len1 + len2 + len3 + len4));

  auto res = LIBC_NAMESPACE::net::if_nameindex<Policy>();
  ASSERT_TRUE(res.has_value());
  struct if_nameindex *list = res.value();
  ASSERT_NE(list, static_cast<struct if_nameindex *>(nullptr));

  ASSERT_EQ(list[0].if_index, 1u);
  ASSERT_STREQ(list[0].if_name, "lo");
  ASSERT_EQ(list[1].if_index, 2u);
  ASSERT_STREQ(list[1].if_name, "eth0");
  ASSERT_EQ(list[2].if_index, 3u);
  ASSERT_STREQ(list[2].if_name, "wlan0");
  ASSERT_EQ(list[3].if_index, 0u);
  ASSERT_EQ(list[3].if_name, static_cast<char *>(nullptr));

  validate_dump_request();

  ASSERT_EQ(policy_data.recv_calls.size(), size_t(1));
  ASSERT_EQ(get<0>(policy_data.recv_calls[0]), FAKE_SOCKET);

  LIBC_NAMESPACE::if_freenameindex(list);
}

TEST_F(LlvmLibcIfNameIndexSocketTest, MultipleRecvCalls) {
  uint8_t pkt_buf[2048];
  size_t len1 = build_ifinfomsg_packet(pkt_buf, 1, AttrName{"lo"});
  size_t len2 = build_ifinfomsg_packet(pkt_buf + len1, 2, AttrName{"eth0"});
  size_t len3 =
      build_ifinfomsg_packet(pkt_buf + len1 + len2, 3, AttrName{"wlan0"});
  size_t len4 = build_nlmsg_done_packet(pkt_buf + len1 + len2 + len3);

  // Two interfaces in the first datagram, one in the second, and NLMSG_DONE in
  // a datagram of its own (which is what the kernel usually does).
  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf, len1 + len2));
  policy_data.recv_results.push_back(
      span<const uint8_t>(pkt_buf + len1 + len2, len3));
  policy_data.recv_results.push_back(
      span<const uint8_t>(pkt_buf + len1 + len2 + len3, len4));

  auto res = LIBC_NAMESPACE::net::if_nameindex<Policy>();
  ASSERT_TRUE(res.has_value());
  struct if_nameindex *list = res.value();
  ASSERT_NE(list, static_cast<struct if_nameindex *>(nullptr));

  ASSERT_EQ(list[0].if_index, 1u);
  ASSERT_STREQ(list[0].if_name, "lo");
  ASSERT_EQ(list[1].if_index, 2u);
  ASSERT_STREQ(list[1].if_name, "eth0");
  ASSERT_EQ(list[2].if_index, 3u);
  ASSERT_STREQ(list[2].if_name, "wlan0");
  ASSERT_EQ(list[3].if_index, 0u);
  ASSERT_EQ(list[3].if_name, static_cast<char *>(nullptr));

  validate_dump_request();

  ASSERT_EQ(policy_data.recv_calls.size(), size_t(3));
  for (size_t i = 0; i < policy_data.recv_calls.size(); ++i)
    ASSERT_EQ(get<0>(policy_data.recv_calls[i]), FAKE_SOCKET);

  LIBC_NAMESPACE::if_freenameindex(list);
}

TEST_F(LlvmLibcIfNameIndexSocketTest, InterruptedDump) {
  // In the first attempt, old0 gets replaced by new0 between the two datagrams,
  // so the kernel marks the second one as interrupted.
  uint8_t pkt_buf1[1024];
  size_t len1 = build_ifinfomsg_packet(pkt_buf1, 1, AttrName{"lo"});
  len1 += build_ifinfomsg_packet(pkt_buf1 + len1, 2, AttrName{"old0"});
  uint8_t pkt_buf2[1024];
  size_t len2 = build_ifinfomsg_packet(pkt_buf2, 3, AttrName{"new0"});
  mark_interrupted(pkt_buf2);
  len2 += build_nlmsg_done_packet(pkt_buf2 + len2);
  // The second attempt gets a consistent view.
  uint8_t pkt_buf3[1024];
  size_t len3 = build_ifinfomsg_packet(pkt_buf3, 1, AttrName{"lo"});
  len3 += build_ifinfomsg_packet(pkt_buf3 + len3, 3, AttrName{"new0"});
  len3 += build_nlmsg_done_packet(pkt_buf3 + len3);

  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf1, len1));
  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf2, len2));
  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf3, len3));

  auto res = LIBC_NAMESPACE::net::if_nameindex<Policy>();
  ASSERT_TRUE(res.has_value());
  struct if_nameindex *list = res.value();
  ASSERT_NE(list, static_cast<struct if_nameindex *>(nullptr));

  ASSERT_EQ(list[0].if_index, 1u);
  ASSERT_STREQ(list[0].if_name, "lo");
  ASSERT_EQ(list[1].if_index, 3u);
  ASSERT_STREQ(list[1].if_name, "new0");
  ASSERT_EQ(list[2].if_index, 0u);
  ASSERT_EQ(list[2].if_name, static_cast<char *>(nullptr));

  validate_dump_request(2);

  // The second attempt uses a larger buffer.
  ASSERT_EQ(policy_data.recv_calls.size(), size_t(3));
  ASSERT_EQ(get<1>(policy_data.recv_calls[0]), Socket::NLMSG_BUFFER_SIZE);
  ASSERT_EQ(get<1>(policy_data.recv_calls[1]), Socket::NLMSG_BUFFER_SIZE);
  ASSERT_EQ(get<1>(policy_data.recv_calls[2]), Socket::NLMSG_LARGE_BUFFER_SIZE);

  LIBC_NAMESPACE::if_freenameindex(list);
}

TEST_F(LlvmLibcIfNameIndexSocketTest, InterruptedDumpUnflaggedDone) {
  // The kernel only marks the first datagram after the change. The rest of the
  // dump, including NLMSG_DONE, arrives unmarked, but we still need to retry.
  uint8_t pkt_buf1[1024];
  size_t len1 = build_ifinfomsg_packet(pkt_buf1, 1, AttrName{"lo"});
  len1 += build_ifinfomsg_packet(pkt_buf1 + len1, 2, AttrName{"old0"});
  uint8_t pkt_buf2[1024];
  size_t len2 = build_ifinfomsg_packet(pkt_buf2, 3, AttrName{"new0"});
  mark_interrupted(pkt_buf2);
  uint8_t done_buf[128];
  size_t done_len = build_nlmsg_done_packet(done_buf);
  // The second attempt gets a consistent view.
  uint8_t pkt_buf3[1024];
  size_t len3 = build_ifinfomsg_packet(pkt_buf3, 1, AttrName{"lo"});
  len3 += build_ifinfomsg_packet(pkt_buf3 + len3, 3, AttrName{"new0"});

  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf1, len1));
  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf2, len2));
  policy_data.recv_results.push_back(span<const uint8_t>(done_buf, done_len));
  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf3, len3));
  policy_data.recv_results.push_back(span<const uint8_t>(done_buf, done_len));

  auto res = LIBC_NAMESPACE::net::if_nameindex<Policy>();
  ASSERT_TRUE(res.has_value());
  struct if_nameindex *list = res.value();
  ASSERT_NE(list, static_cast<struct if_nameindex *>(nullptr));

  ASSERT_EQ(list[0].if_index, 1u);
  ASSERT_STREQ(list[0].if_name, "lo");
  ASSERT_EQ(list[1].if_index, 3u);
  ASSERT_STREQ(list[1].if_name, "new0");
  ASSERT_EQ(list[2].if_index, 0u);
  ASSERT_EQ(list[2].if_name, static_cast<char *>(nullptr));

  validate_dump_request(2);

  // Both datagrams of the second attempt are received into the larger buffer.
  ASSERT_EQ(policy_data.recv_calls.size(), size_t(5));
  for (size_t i = 0; i < 3; ++i)
    ASSERT_EQ(get<1>(policy_data.recv_calls[i]), Socket::NLMSG_BUFFER_SIZE);
  for (size_t i = 3; i < 5; ++i)
    ASSERT_EQ(get<1>(policy_data.recv_calls[i]),
              Socket::NLMSG_LARGE_BUFFER_SIZE);

  LIBC_NAMESPACE::if_freenameindex(list);
}

TEST_F(LlvmLibcIfNameIndexSocketTest, InterruptedDumpDone) {
  // The interruption can also be reported on the NLMSG_DONE message.
  uint8_t pkt_buf1[1024];
  size_t len1 = build_ifinfomsg_packet(pkt_buf1, 1, AttrName{"lo"});
  uint8_t done_buf[128];
  size_t done_len = build_nlmsg_done_packet(done_buf);
  mark_interrupted(done_buf);
  uint8_t pkt_buf2[1024];
  size_t len2 = build_ifinfomsg_packet(pkt_buf2, 2, AttrName{"eth0"});
  len2 += build_nlmsg_done_packet(pkt_buf2 + len2);

  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf1, len1));
  policy_data.recv_results.push_back(span<const uint8_t>(done_buf, done_len));
  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf2, len2));

  auto res = LIBC_NAMESPACE::net::if_nameindex<Policy>();
  ASSERT_TRUE(res.has_value());
  struct if_nameindex *list = res.value();
  ASSERT_NE(list, static_cast<struct if_nameindex *>(nullptr));

  ASSERT_EQ(list[0].if_index, 2u);
  ASSERT_STREQ(list[0].if_name, "eth0");
  ASSERT_EQ(list[1].if_index, 0u);
  ASSERT_EQ(list[1].if_name, static_cast<char *>(nullptr));

  validate_dump_request(2);
  LIBC_NAMESPACE::if_freenameindex(list);
}

TEST_F(LlvmLibcIfNameIndexSocketTest, TooManyInterruptions) {
  uint8_t pkt_buf[1024];
  size_t len = build_ifinfomsg_packet(pkt_buf, 1, AttrName{"lo"});
  mark_interrupted(pkt_buf);
  len += build_nlmsg_done_packet(pkt_buf + len);

  // Every attempt gets interrupted.
  for (size_t i = 0; i < MAX_DUMP_ATTEMPTS; ++i)
    policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf, len));

  auto res = LIBC_NAMESPACE::net::if_nameindex<Policy>();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), EAGAIN);
  validate_dump_request(MAX_DUMP_ATTEMPTS);

  // We switch to the larger buffer after the first attempt and keep using it.
  ASSERT_EQ(policy_data.recv_calls.size(), MAX_DUMP_ATTEMPTS);
  ASSERT_EQ(get<1>(policy_data.recv_calls[0]), Socket::NLMSG_BUFFER_SIZE);
  for (size_t i = 1; i < MAX_DUMP_ATTEMPTS; ++i)
    ASSERT_EQ(get<1>(policy_data.recv_calls[i]),
              Socket::NLMSG_LARGE_BUFFER_SIZE);
}

TEST_F(LlvmLibcIfNameIndexSocketTest, RecvFailure) {
  policy_data.recv_results.push_back(Error(ETIMEDOUT));

  auto res = LIBC_NAMESPACE::net::if_nameindex<Policy>();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), ETIMEDOUT);
  validate_dump_request();
}

TEST_F(LlvmLibcIfNameIndexSocketTest, SecondRecvFailure) {
  uint8_t pkt_buf[1024];
  size_t len = build_ifinfomsg_packet(pkt_buf, 1, AttrName{"lo"});

  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf, len));
  policy_data.recv_results.push_back(Error(ENOMEM));

  auto res = LIBC_NAMESPACE::net::if_nameindex<Policy>();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), ENOMEM);
  validate_dump_request();
}

TEST_F(LlvmLibcIfNameIndexSocketTest, ZeroLengthRecv) {
  uint8_t pkt_buf[1024];
  size_t len = build_ifinfomsg_packet(pkt_buf, 1, AttrName{"lo"});

  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf, len));
  policy_data.recv_results.push_back(span<const uint8_t>());

  auto res = LIBC_NAMESPACE::net::if_nameindex<Policy>();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), EIO);
  validate_dump_request();
}

TEST_F(LlvmLibcIfNameIndexSocketTest, NetlinkErrorPacket) {
  uint8_t pkt_buf[128];
  size_t len = build_nlmsg_error_packet(pkt_buf, -EINVAL);

  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf, len));

  auto res = LIBC_NAMESPACE::net::if_nameindex<Policy>();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), EINVAL);
  validate_dump_request();
}

TEST_F(LlvmLibcIfNameIndexSocketTest, TruncatedNetlinkErrorPacket) {
  uint8_t pkt_buf[128];
  size_t len = build_generic_nlmsg_packet(pkt_buf, NLMSG_ERROR);

  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf, len));

  auto res = LIBC_NAMESPACE::net::if_nameindex<Policy>();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), EINVAL);
  validate_dump_request();
}

TEST_F(LlvmLibcIfNameIndexSocketTest, NetlinkDoneWithError) {
  uint8_t pkt_buf[1024];
  size_t len1 = build_ifinfomsg_packet(pkt_buf, 1, AttrName{"lo"});
  size_t len2 = build_nlmsg_done_packet(pkt_buf + len1, -EMSGSIZE);

  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf, len1));
  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf + len1, len2));

  auto res = LIBC_NAMESPACE::net::if_nameindex<Policy>();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), EMSGSIZE);
  validate_dump_request();
}

TEST_F(LlvmLibcIfNameIndexSocketTest, TruncatedNetlinkDonePacket) {
  uint8_t pkt_buf[128];
  size_t len = build_generic_nlmsg_packet(pkt_buf, NLMSG_DONE);

  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf, len));

  auto res = LIBC_NAMESPACE::net::if_nameindex<Policy>();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), EIO);
  validate_dump_request();
}

TEST_F(LlvmLibcIfNameIndexSocketTest, TruncatedIfInfoMsgPacket) {
  uint8_t pkt_buf[1024];
  size_t len1 = build_generic_nlmsg_packet(pkt_buf, RTM_NEWLINK);
  size_t len2 = build_ifinfomsg_packet(pkt_buf + len1, 2, AttrName{"eth0"});
  size_t len3 = build_nlmsg_done_packet(pkt_buf + len1 + len2);

  policy_data.recv_results.push_back(
      span<const uint8_t>(pkt_buf, len1 + len2 + len3));

  auto res = LIBC_NAMESPACE::net::if_nameindex<Policy>();
  ASSERT_TRUE(res.has_value());
  struct if_nameindex *list = res.value();
  ASSERT_NE(list, static_cast<struct if_nameindex *>(nullptr));

  ASSERT_EQ(list[0].if_index, 2u);
  ASSERT_STREQ(list[0].if_name, "eth0");
  ASSERT_EQ(list[1].if_index, 0u);
  ASSERT_EQ(list[1].if_name, static_cast<char *>(nullptr));

  validate_dump_request();
  LIBC_NAMESPACE::if_freenameindex(list);
}

TEST_F(LlvmLibcIfNameIndexSocketTest, TruncatedNetlinkHeaderPacket) {
  uint8_t pkt_buf[4] = {1, 2, 3, 4};
  policy_data.recv_results.push_back(
      span<const uint8_t>(pkt_buf, sizeof(pkt_buf)));

  auto res = LIBC_NAMESPACE::net::if_nameindex<Policy>();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), EIO);
  validate_dump_request();
  ASSERT_EQ(policy_data.recv_calls.size(), size_t(1));
}

TEST_F(LlvmLibcIfNameIndexSocketTest, TruncatedDatagram) {
  uint8_t pkt_buf[1024];
  size_t len1 = build_ifinfomsg_packet(pkt_buf, 1, AttrName{"lo"});
  size_t len2 = build_ifinfomsg_packet(pkt_buf + len1, 2, AttrName{"eth0"});

  // Cut off the end of the second message.
  policy_data.recv_results.push_back(
      span<const uint8_t>(pkt_buf, len1 + len2 - 4));

  auto res = LIBC_NAMESPACE::net::if_nameindex<Policy>();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), EIO);
  validate_dump_request();
  ASSERT_EQ(policy_data.recv_calls.size(), size_t(1));
}

TEST_F(LlvmLibcIfNameIndexSocketTest, CloseFailure) {
  uint8_t pkt_buf[128];
  size_t len = build_nlmsg_done_packet(pkt_buf);

  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf, len));
  policy_data.close_results.push_back(Error(EIO));

  auto res = LIBC_NAMESPACE::net::if_nameindex<Policy>();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), EIO);
  validate_dump_request();
  ASSERT_EQ(policy_data.recv_calls.size(), size_t(1));
}

TEST_F(LlvmLibcIfNameIndexSocketTest, NetlinkAckPacket) {
  uint8_t pkt_buf[1024];
  size_t len1 = build_nlmsg_error_packet(pkt_buf, 0);
  size_t len2 = build_ifinfomsg_packet(pkt_buf + len1, 1, AttrName{"lo"});
  size_t len3 = build_nlmsg_done_packet(pkt_buf + len1 + len2);

  policy_data.recv_results.push_back(
      span<const uint8_t>(pkt_buf, len1 + len2 + len3));

  auto res = LIBC_NAMESPACE::net::if_nameindex<Policy>();
  ASSERT_TRUE(res.has_value());
  struct if_nameindex *list = res.value();
  ASSERT_NE(list, static_cast<struct if_nameindex *>(nullptr));

  ASSERT_EQ(list[0].if_index, 1u);
  ASSERT_STREQ(list[0].if_name, "lo");
  ASSERT_EQ(list[1].if_index, 0u);
  ASSERT_EQ(list[1].if_name, static_cast<char *>(nullptr));

  validate_dump_request();
  LIBC_NAMESPACE::if_freenameindex(list);
}

TEST_F(LlvmLibcIfNameIndexSocketTest, IgnoredNetlinkMessageTypes) {
  uint8_t pkt_buf[1024];
  size_t len1 = build_generic_nlmsg_packet(pkt_buf, NLMSG_NOOP);
  size_t len2 = build_ifinfomsg_packet(pkt_buf + len1, 2, AttrName{"eth0"});
  size_t len3 = build_nlmsg_done_packet(pkt_buf + len1 + len2);

  policy_data.recv_results.push_back(
      span<const uint8_t>(pkt_buf, len1 + len2 + len3));

  auto res = LIBC_NAMESPACE::net::if_nameindex<Policy>();
  ASSERT_TRUE(res.has_value());
  struct if_nameindex *list = res.value();
  ASSERT_NE(list, static_cast<struct if_nameindex *>(nullptr));

  ASSERT_EQ(list[0].if_index, 2u);
  ASSERT_STREQ(list[0].if_name, "eth0");
  ASSERT_EQ(list[1].if_index, 0u);
  ASSERT_EQ(list[1].if_name, static_cast<char *>(nullptr));

  validate_dump_request();
  LIBC_NAMESPACE::if_freenameindex(list);
}

TEST_F(LlvmLibcIfNameIndexSocketTest, InterfaceWithMultipleAttributes) {
  uint8_t pkt_buf[1024];
  size_t len1 = build_ifinfomsg_packet(pkt_buf, 3, AttrMtu{1500},
                                       AttrName{"wlan0"}, AttrTxqlen{1000});
  size_t len2 = build_nlmsg_done_packet(pkt_buf + len1);

  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf, len1 + len2));

  auto res = LIBC_NAMESPACE::net::if_nameindex<Policy>();
  ASSERT_TRUE(res.has_value());
  struct if_nameindex *list = res.value();
  ASSERT_NE(list, static_cast<struct if_nameindex *>(nullptr));

  ASSERT_EQ(list[0].if_index, 3u);
  ASSERT_STREQ(list[0].if_name, "wlan0");
  ASSERT_EQ(list[1].if_index, 0u);
  ASSERT_EQ(list[1].if_name, static_cast<char *>(nullptr));

  validate_dump_request();
  LIBC_NAMESPACE::if_freenameindex(list);
}

TEST_F(LlvmLibcIfNameIndexSocketTest, InterfaceMissingNameAttribute) {
  uint8_t pkt_buf[1024];
  size_t len1 = build_ifinfomsg_packet(pkt_buf, 1, AttrMtu{1500});
  size_t len2 = build_ifinfomsg_packet(pkt_buf + len1, 2, AttrName{"eth0"});
  size_t len3 = build_nlmsg_done_packet(pkt_buf + len1 + len2);

  policy_data.recv_results.push_back(
      span<const uint8_t>(pkt_buf, len1 + len2 + len3));

  auto res = LIBC_NAMESPACE::net::if_nameindex<Policy>();
  ASSERT_TRUE(res.has_value());
  struct if_nameindex *list = res.value();
  ASSERT_NE(list, static_cast<struct if_nameindex *>(nullptr));

  ASSERT_EQ(list[0].if_index, 2u);
  ASSERT_STREQ(list[0].if_name, "eth0");
  ASSERT_EQ(list[1].if_index, 0u);
  ASSERT_EQ(list[1].if_name, static_cast<char *>(nullptr));

  validate_dump_request();
  LIBC_NAMESPACE::if_freenameindex(list);
}

TEST_F(LlvmLibcIfNameIndexSocketTest, InterfaceNameWithoutNullTerminator) {
  uint8_t pkt_buf[1024];
  size_t len1 = build_ifinfomsg_packet(pkt_buf, 4, AttrName{"docker0", false});
  size_t len2 = build_nlmsg_done_packet(pkt_buf + len1);

  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf, len1 + len2));

  auto res = LIBC_NAMESPACE::net::if_nameindex<Policy>();
  ASSERT_TRUE(res.has_value());
  struct if_nameindex *list = res.value();
  ASSERT_NE(list, static_cast<struct if_nameindex *>(nullptr));

  ASSERT_EQ(list[0].if_index, 4u);
  ASSERT_STREQ(list[0].if_name, "docker0");
  ASSERT_EQ(list[1].if_index, 0u);
  ASSERT_EQ(list[1].if_name, static_cast<char *>(nullptr));

  validate_dump_request();
  LIBC_NAMESPACE::if_freenameindex(list);
}

TEST_F(LlvmLibcIfNameIndexSocketTest, InterfaceNameExceedingIfNamesize) {
  uint8_t pkt_buf[1024];
  size_t len1 = build_ifinfomsg_packet(
      pkt_buf, 5,
      AttrName{"this_interface_name_is_way_too_long_for_if_namesize"});
  size_t len2 = build_nlmsg_done_packet(pkt_buf + len1);

  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf, len1 + len2));

  auto res = LIBC_NAMESPACE::net::if_nameindex<Policy>();
  ASSERT_TRUE(res.has_value());
  struct if_nameindex *list = res.value();
  ASSERT_NE(list, static_cast<struct if_nameindex *>(nullptr));

  ASSERT_EQ(list[0].if_index, 5u);
  ASSERT_STREQ(list[0].if_name, "this_interface_");
  ASSERT_EQ(list[1].if_index, 0u);
  ASSERT_EQ(list[1].if_name, static_cast<char *>(nullptr));

  validate_dump_request();
  LIBC_NAMESPACE::if_freenameindex(list);
}

using LlvmLibcIfNameIndexLiveTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcIfNameIndexLiveTest, LiveOSIntegration) {
  struct if_nameindex *list = LIBC_NAMESPACE::if_nameindex();
  ASSERT_NE(list, static_cast<struct if_nameindex *>(nullptr));
  ASSERT_ERRNO_SUCCESS();

  bool found_valid = false;
  for (struct if_nameindex *cur = list;
       cur->if_index != 0 || cur->if_name != nullptr; ++cur) {
    ASSERT_GT(cur->if_index, 0u);
    ASSERT_NE(cur->if_name, static_cast<char *>(nullptr));
    if (LIBC_NAMESPACE::cpp::string_view(cur->if_name) == "lo" ||
        cur->if_index > 0)
      found_valid = true;
  }
  ASSERT_TRUE(found_valid);

  LIBC_NAMESPACE::if_freenameindex(list);
}

TEST_F(LlvmLibcIfNameIndexLiveTest, FreeNullptr) {
  LIBC_NAMESPACE::if_freenameindex(nullptr);
}
