//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for getifaddrs and freeifaddrs.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/sys_resource_macros.h"
#include "hdr/sys_socket_macros.h"
#include "hdr/types/sa_family_t.h"
#include "hdr/types/struct_ifaddrs.h"
#include "hdr/types/struct_rlimit.h"
#include "hdr/types/struct_sockaddr.h"
#include "src/__support/CPP/scope.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/CPP/tuple.h"
#include "src/__support/error_or.h"
#include "src/__support/net/linux/netlink.h"
#include "src/ifaddrs/freeifaddrs.h"
#include "src/ifaddrs/getifaddrs.h"
#include "src/ifaddrs/linux/getifaddrs_impl.h"
#include "src/sys/resource/getrlimit.h"
#include "src/sys/resource/setrlimit.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"
#include "test/src/__support/net/linux/netlink_test_support.h"

#include <linux/if_link.h>
#include <linux/if_packet.h>
#include <linux/netdevice.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LIBC_NAMESPACE::Error;
using LIBC_NAMESPACE::cpp::get;
using LIBC_NAMESPACE::cpp::span;
using LIBC_NAMESPACE::cpp::string_view;
using LIBC_NAMESPACE::net::MAX_DUMP_ATTEMPTS;
using LIBC_NAMESPACE::net::detail::SockAddrLL;
using LIBC_NAMESPACE::testing::AttrBytes;
using LIBC_NAMESPACE::testing::AttrString;
using LIBC_NAMESPACE::testing::build_generic_nlmsg_packet;
using LIBC_NAMESPACE::testing::build_nlmsg_done_packet;
using LIBC_NAMESPACE::testing::build_nlmsg_packet;
using LIBC_NAMESPACE::testing::FakeNetworkSyscallPolicy;
using LIBC_NAMESPACE::testing::FakeNetworkSyscallPolicyData;
using LIBC_NAMESPACE::testing::mark_interrupted;

namespace {

// Hardware types (ARPHRD_*) and interface flags (IFF_*). We just check that
// these get passed through, so their meaning is not important.
constexpr unsigned short HW_TYPE_ETHER = 1;
constexpr unsigned short HW_TYPE_INFINIBAND = 32;
constexpr unsigned short HW_TYPE_NONE = 65534;
constexpr unsigned int FLAGS_ETH = 0x11043;
constexpr unsigned int FLAGS_LO = 0x10049;
constexpr unsigned int FLAGS_WG = 0x91;

using AttrName = AttrString<IFLA_IFNAME>;
using AttrAddress = AttrBytes<IFLA_ADDRESS>;
using AttrBroadcast = AttrBytes<IFLA_BROADCAST>;

constexpr uint8_t ETH0_ADDR[] = {0x02, 0x42, 0xac, 0x11, 0x00, 0x02};
constexpr uint8_t ETH_BROADCAST[] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
constexpr uint8_t LO_ADDR[] = {0, 0, 0, 0, 0, 0};

struct LlvmLibcGetIfAddrsTest
    : public LIBC_NAMESPACE::testing::ErrnoCheckingTest {
  static constexpr int FAKE_SOCKET = 47;
  static FakeNetworkSyscallPolicyData policy_data;
  using Policy = FakeNetworkSyscallPolicy<&policy_data>;

  void SetUp() override {
    ErrnoCheckingTest::SetUp();
    policy_data = {};
  }
};

struct LlvmLibcGetIfAddrsSocketTest : public LlvmLibcGetIfAddrsTest {
  void SetUp() override {
    LlvmLibcGetIfAddrsTest::SetUp();
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
    LlvmLibcGetIfAddrsTest::TearDown();
  }

  /// Checks that we sent the expected number of RTM_GETLINK requests.
  void validate_link_requests(size_t num_requests = 1) {
    constexpr size_t REQUEST_LEN =
        static_cast<size_t>(NLMSG_LENGTH(sizeof(ifinfomsg)));
    ASSERT_EQ(policy_data.sendto_calls.size(), num_requests);
    ASSERT_EQ(policy_data.sendto_data.size(), num_requests * REQUEST_LEN);
    for (size_t i = 0; i < num_requests; ++i) {
      ASSERT_EQ(get<0>(policy_data.sendto_calls[i]), FAKE_SOCKET);
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

} // namespace

FakeNetworkSyscallPolicyData LlvmLibcGetIfAddrsTest::policy_data;

template <typename... Attrs>
static size_t build_link_packet(uint8_t *buf, int index, unsigned int flags,
                                unsigned short type, const Attrs &...attrs) {
  struct ifinfomsg ifm = {};
  ifm.ifi_family = AF_UNSPEC;
  ifm.ifi_type = type;
  ifm.ifi_index = index;
  ifm.ifi_flags = flags;
  return build_nlmsg_packet(buf, RTM_NEWLINK, ifm, attrs...);
}

static string_view as_string_view(span<const uint8_t> bytes) {
  return string_view(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

/// Returns the hardware address stored in an AF_PACKET address.
static string_view get_link_address(const struct sockaddr *addr) {
  auto *ll = reinterpret_cast<const SockAddrLL *>(addr);
  return string_view(reinterpret_cast<const char *>(ll->sll_addr),
                     ll->sll_halen);
}

TEST_F(LlvmLibcGetIfAddrsTest, SocketCreationFailure) {
  policy_data.socket_results.push_back(Error(EMFILE));

  auto res = LIBC_NAMESPACE::net::getifaddrs<Policy>();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), EMFILE);
  ASSERT_EQ(policy_data.close_calls.size(), size_t(0));
}

TEST_F(LlvmLibcGetIfAddrsSocketTest, SendFailure) {
  policy_data.sendto_results.push_back(Error(ENOBUFS));

  auto res = LIBC_NAMESPACE::net::getifaddrs<Policy>();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), ENOBUFS);
  validate_link_requests();
}

TEST_F(LlvmLibcGetIfAddrsSocketTest, RecvFailure) {
  policy_data.recv_results.push_back(Error(EINTR));

  auto res = LIBC_NAMESPACE::net::getifaddrs<Policy>();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), EINTR);
  validate_link_requests();
}

TEST_F(LlvmLibcGetIfAddrsSocketTest, CloseFailure) {
  alignas(struct nlmsghdr) uint8_t pkt_buf[1024];
  size_t len =
      build_link_packet(pkt_buf, 1, FLAGS_LO, HW_TYPE_ETHER, AttrName{"lo"});
  len += build_nlmsg_done_packet(pkt_buf + len);
  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf, len));
  policy_data.close_results.push_back(Error(EIO));

  auto res = LIBC_NAMESPACE::net::getifaddrs<Policy>();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), EIO);
  validate_link_requests();
}

TEST_F(LlvmLibcGetIfAddrsSocketTest, NoInterfaces) {
  alignas(struct nlmsghdr) uint8_t pkt_buf[128];
  size_t len = build_nlmsg_done_packet(pkt_buf);
  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf, len));

  auto res = LIBC_NAMESPACE::net::getifaddrs<Policy>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(*res, static_cast<struct ifaddrs *>(nullptr));
  validate_link_requests();
}

TEST_F(LlvmLibcGetIfAddrsSocketTest, SingleInterface) {
  alignas(struct nlmsghdr) uint8_t pkt_buf[1024];
  size_t len =
      build_link_packet(pkt_buf, 2, FLAGS_ETH, HW_TYPE_ETHER, AttrName{"eth0"},
                        AttrAddress{ETH0_ADDR}, AttrBroadcast{ETH_BROADCAST});
  len += build_nlmsg_done_packet(pkt_buf + len);
  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf, len));

  auto res = LIBC_NAMESPACE::net::getifaddrs<Policy>();
  ASSERT_TRUE(res.has_value());
  struct ifaddrs *list = *res;
  ASSERT_NE(list, static_cast<struct ifaddrs *>(nullptr));

  ASSERT_STREQ(list->ifa_name, "eth0");
  ASSERT_EQ(list->ifa_flags, FLAGS_ETH);
  ASSERT_NE(list->ifa_addr, static_cast<struct sockaddr *>(nullptr));
  auto *ll = reinterpret_cast<struct sockaddr_ll *>(list->ifa_addr);
  ASSERT_EQ(ll->sll_family, static_cast<unsigned short>(AF_PACKET));
  ASSERT_EQ(ll->sll_ifindex, 2);
  ASSERT_EQ(ll->sll_hatype, HW_TYPE_ETHER);
  ASSERT_EQ(get_link_address(list->ifa_addr), as_string_view(ETH0_ADDR));
  ASSERT_EQ(list->ifa_netmask, static_cast<struct sockaddr *>(nullptr));
  ASSERT_NE(list->ifa_broadaddr, static_cast<struct sockaddr *>(nullptr));
  ASSERT_EQ(list->ifa_broadaddr->sa_family,
            static_cast<sa_family_t>(AF_PACKET));
  ASSERT_EQ(get_link_address(list->ifa_broadaddr),
            as_string_view(ETH_BROADCAST));
  ASSERT_EQ(list->ifa_data, static_cast<void *>(nullptr));
  ASSERT_EQ(list->ifa_next, static_cast<struct ifaddrs *>(nullptr));

  validate_link_requests();
  LIBC_NAMESPACE::freeifaddrs(list);
}

TEST_F(LlvmLibcGetIfAddrsSocketTest, MultipleInterfaces) {
  alignas(struct nlmsghdr) uint8_t pkt_buf[2048];
  size_t len1 =
      build_link_packet(pkt_buf, 1, FLAGS_LO, HW_TYPE_ETHER, AttrName{"lo"},
                        AttrAddress{LO_ADDR}, AttrBroadcast{LO_ADDR});
  len1 += build_link_packet(pkt_buf + len1, 2, FLAGS_ETH, HW_TYPE_ETHER,
                            AttrName{"eth0"}, AttrAddress{ETH0_ADDR},
                            AttrBroadcast{ETH_BROADCAST});
  // Interfaces without a hardware address (e.g. wireguard) don't get the
  // address attributes.
  size_t len2 = build_link_packet(pkt_buf + len1, 3, FLAGS_WG, HW_TYPE_NONE,
                                  AttrName{"wg0"});
  len2 += build_nlmsg_done_packet(pkt_buf + len1 + len2);

  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf, len1));
  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf + len1, len2));

  auto res = LIBC_NAMESPACE::net::getifaddrs<Policy>();
  ASSERT_TRUE(res.has_value());
  struct ifaddrs *list = *res;

  struct ifaddrs *ifa = list;
  ASSERT_NE(ifa, static_cast<struct ifaddrs *>(nullptr));
  ASSERT_STREQ(ifa->ifa_name, "lo");
  ASSERT_EQ(ifa->ifa_flags, FLAGS_LO);
  ASSERT_NE(ifa->ifa_addr, static_cast<struct sockaddr *>(nullptr));
  ASSERT_EQ(get_link_address(ifa->ifa_addr), as_string_view(LO_ADDR));

  ifa = ifa->ifa_next;
  ASSERT_NE(ifa, static_cast<struct ifaddrs *>(nullptr));
  ASSERT_STREQ(ifa->ifa_name, "eth0");
  ASSERT_EQ(ifa->ifa_flags, FLAGS_ETH);
  ASSERT_NE(ifa->ifa_addr, static_cast<struct sockaddr *>(nullptr));
  ASSERT_EQ(get_link_address(ifa->ifa_addr), as_string_view(ETH0_ADDR));

  ifa = ifa->ifa_next;
  ASSERT_NE(ifa, static_cast<struct ifaddrs *>(nullptr));
  ASSERT_STREQ(ifa->ifa_name, "wg0");
  ASSERT_EQ(ifa->ifa_flags, FLAGS_WG);
  ASSERT_EQ(ifa->ifa_addr, static_cast<struct sockaddr *>(nullptr));
  ASSERT_EQ(ifa->ifa_broadaddr, static_cast<struct sockaddr *>(nullptr));

  ASSERT_EQ(ifa->ifa_next, static_cast<struct ifaddrs *>(nullptr));

  validate_link_requests();
  ASSERT_EQ(policy_data.recv_calls.size(), size_t(2));
  LIBC_NAMESPACE::freeifaddrs(list);
}

TEST_F(LlvmLibcGetIfAddrsSocketTest, LongHardwareAddresses) {
  // SockAddrLL has room for up to MAX_ADDR_LEN (32) bytes (covering e.g.
  // 20-byte InfiniBand addresses), which doesn't fit into sockaddr_ll. Anything
  // longer is ignored.
  uint8_t max_addr[MAX_ADDR_LEN];
  uint8_t too_long_addr[MAX_ADDR_LEN + 1];
  for (size_t i = 0; i < sizeof(too_long_addr); ++i) {
    too_long_addr[i] = static_cast<uint8_t>(i);
    if (i < sizeof(max_addr))
      max_addr[i] = static_cast<uint8_t>(0x80 + i);
  }
  alignas(struct nlmsghdr) uint8_t pkt_buf[1024];
  size_t len = build_link_packet(pkt_buf, 4, FLAGS_ETH, HW_TYPE_INFINIBAND,
                                 AttrName{"ib0"}, AttrAddress{max_addr},
                                 AttrBroadcast{too_long_addr});
  len += build_nlmsg_done_packet(pkt_buf + len);
  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf, len));

  auto res = LIBC_NAMESPACE::net::getifaddrs<Policy>();
  ASSERT_TRUE(res.has_value());
  struct ifaddrs *list = *res;
  ASSERT_NE(list, static_cast<struct ifaddrs *>(nullptr));

  ASSERT_STREQ(list->ifa_name, "ib0");
  ASSERT_NE(list->ifa_addr, static_cast<struct sockaddr *>(nullptr));
  auto *ll = reinterpret_cast<const SockAddrLL *>(list->ifa_addr);
  ASSERT_EQ(ll->sll_hatype, HW_TYPE_INFINIBAND);
  ASSERT_EQ(get_link_address(list->ifa_addr), as_string_view(max_addr));
  ASSERT_EQ(list->ifa_broadaddr, static_cast<struct sockaddr *>(nullptr));

  validate_link_requests();
  LIBC_NAMESPACE::freeifaddrs(list);
}

TEST_F(LlvmLibcGetIfAddrsSocketTest, InterfaceWithoutName) {
  alignas(struct nlmsghdr) uint8_t pkt_buf[1024];
  size_t len = build_link_packet(pkt_buf, 1, FLAGS_ETH, HW_TYPE_ETHER,
                                 AttrAddress{ETH0_ADDR});
  len += build_link_packet(pkt_buf + len, 2, FLAGS_ETH, HW_TYPE_ETHER,
                           AttrName{"eth1"});
  len += build_nlmsg_done_packet(pkt_buf + len);
  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf, len));

  auto res = LIBC_NAMESPACE::net::getifaddrs<Policy>();
  ASSERT_TRUE(res.has_value());
  struct ifaddrs *list = *res;
  ASSERT_NE(list, static_cast<struct ifaddrs *>(nullptr));
  ASSERT_STREQ(list->ifa_name, "eth1");
  ASSERT_EQ(list->ifa_next, static_cast<struct ifaddrs *>(nullptr));

  validate_link_requests();
  LIBC_NAMESPACE::freeifaddrs(list);
}

TEST_F(LlvmLibcGetIfAddrsSocketTest, TruncatedIfInfoMsgPacket) {
  alignas(struct nlmsghdr) uint8_t pkt_buf[1024];
  size_t len = build_generic_nlmsg_packet(pkt_buf, RTM_NEWLINK);
  len += build_link_packet(pkt_buf + len, 2, FLAGS_ETH, HW_TYPE_ETHER,
                           AttrName{"eth0"});
  len += build_nlmsg_done_packet(pkt_buf + len);
  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf, len));

  auto res = LIBC_NAMESPACE::net::getifaddrs<Policy>();
  ASSERT_TRUE(res.has_value());
  struct ifaddrs *list = *res;
  ASSERT_NE(list, static_cast<struct ifaddrs *>(nullptr));
  ASSERT_STREQ(list->ifa_name, "eth0");
  ASSERT_EQ(list->ifa_next, static_cast<struct ifaddrs *>(nullptr));

  validate_link_requests();
  LIBC_NAMESPACE::freeifaddrs(list);
}

TEST_F(LlvmLibcGetIfAddrsSocketTest, InterruptedDump) {
  alignas(struct nlmsghdr) uint8_t pkt_buf1[1024];
  size_t len1 = build_link_packet(pkt_buf1, 2, FLAGS_ETH, HW_TYPE_ETHER,
                                  AttrName{"old0"});
  mark_interrupted(pkt_buf1);
  len1 += build_nlmsg_done_packet(pkt_buf1 + len1);
  alignas(struct nlmsghdr) uint8_t pkt_buf2[1024];
  size_t len2 = build_link_packet(pkt_buf2, 3, FLAGS_ETH, HW_TYPE_ETHER,
                                  AttrName{"new0"});
  len2 += build_nlmsg_done_packet(pkt_buf2 + len2);

  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf1, len1));
  policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf2, len2));

  auto res = LIBC_NAMESPACE::net::getifaddrs<Policy>();
  ASSERT_TRUE(res.has_value());
  struct ifaddrs *list = *res;
  ASSERT_NE(list, static_cast<struct ifaddrs *>(nullptr));
  ASSERT_STREQ(list->ifa_name, "new0");
  ASSERT_EQ(list->ifa_next, static_cast<struct ifaddrs *>(nullptr));

  validate_link_requests(2);
  LIBC_NAMESPACE::freeifaddrs(list);
}

TEST_F(LlvmLibcGetIfAddrsSocketTest, TooManyInterruptions) {
  alignas(struct nlmsghdr) uint8_t pkt_buf[1024];
  size_t len =
      build_link_packet(pkt_buf, 1, FLAGS_LO, HW_TYPE_ETHER, AttrName{"lo"});
  mark_interrupted(pkt_buf);
  len += build_nlmsg_done_packet(pkt_buf + len);
  for (size_t i = 0; i < MAX_DUMP_ATTEMPTS; ++i)
    policy_data.recv_results.push_back(span<const uint8_t>(pkt_buf, len));

  auto res = LIBC_NAMESPACE::net::getifaddrs<Policy>();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), EAGAIN);
  validate_link_requests(MAX_DUMP_ATTEMPTS);
}

using LlvmLibcGetIfAddrsLiveTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcGetIfAddrsLiveTest, LiveOSIntegration) {
  struct ifaddrs *list = nullptr;
  ASSERT_THAT(LIBC_NAMESPACE::getifaddrs(&list), Succeeds(0));

  bool found_lo = false;
  for (struct ifaddrs *ifa = list; ifa; ifa = ifa->ifa_next) {
    ASSERT_NE(ifa->ifa_name, static_cast<char *>(nullptr));
    if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_PACKET &&
        string_view(ifa->ifa_name) == "lo")
      found_lo = true;
  }
  // Every network namespace has a loopback interface.
  ASSERT_TRUE(found_lo);

  LIBC_NAMESPACE::freeifaddrs(list);
}

TEST_F(LlvmLibcGetIfAddrsLiveTest, EntrypointFailure) {
  struct rlimit orig_limits;
  ASSERT_THAT(LIBC_NAMESPACE::getrlimit(RLIMIT_NOFILE, &orig_limits),
              Succeeds(0));
  LIBC_NAMESPACE::cpp::scope_exit restore_limits([&] {
    EXPECT_THAT(LIBC_NAMESPACE::setrlimit(RLIMIT_NOFILE, &orig_limits),
                Succeeds(0));
  });

  struct rlimit zero_limits = {0, orig_limits.rlim_max};
  ASSERT_THAT(LIBC_NAMESPACE::setrlimit(RLIMIT_NOFILE, &zero_limits),
              Succeeds(0));

  struct ifaddrs sentinel = {};
  struct ifaddrs *list = &sentinel;
  ASSERT_THAT(LIBC_NAMESPACE::getifaddrs(&list), Fails(EMFILE));
  ASSERT_EQ(list, &sentinel);
}

TEST_F(LlvmLibcGetIfAddrsLiveTest, FreeNullptr) {
  LIBC_NAMESPACE::freeifaddrs(nullptr);
}

#if defined(LIBC_ADD_NULL_CHECKS)
TEST_F(LlvmLibcGetIfAddrsLiveTest, NullPtrDeath) {
  ASSERT_DEATH([]() { LIBC_NAMESPACE::getifaddrs(nullptr); }, WITH_SIGNAL(-1));
}
#endif // defined(LIBC_ADD_NULL_CHECKS)
