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
#include "hdr/types/socklen_t.h"
#include "hdr/types/ssize_t.h"
#include "hdr/types/struct_if_nameindex.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/CPP/tuple.h"
#include "src/__support/CPP/type_traits/type_identity.h"
#include "src/__support/error_or.h"
#include "src/__support/fixedvector.h"
#include "src/net/if_freenameindex.h"
#include "src/net/if_nameindex.h"
#include "src/net/linux/if_nameindex_impl.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/string/memory_utils/inline_memset.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

#include <linux/netlink.h>
#include <linux/rtnetlink.h>

using LIBC_NAMESPACE::Error;
using LIBC_NAMESPACE::ErrorOr;
using LIBC_NAMESPACE::FixedVector;
using LIBC_NAMESPACE::cpp::get;
using LIBC_NAMESPACE::cpp::span;
using LIBC_NAMESPACE::cpp::string;
using LIBC_NAMESPACE::cpp::string_view;
using LIBC_NAMESPACE::cpp::tuple;

// TODO: Add optional::value_or, then return optional<T>.
template <typename T, size_t CAPACITY>
static T
pop_front_or(FixedVector<T, CAPACITY> &vec,
             typename LIBC_NAMESPACE::cpp::type_identity<T>::type default_val) {
  if (vec.empty())
    return default_val;
  // TODO: Add front() and erase() to FixedVector, then clean this up.
  T first = vec[0];
  for (size_t i = 1; i < vec.size(); ++i)
    vec[i - 1] = vec[i];
  vec.pop_back();
  return first;
}

// TODO: Add string::operator+=(string_view), then remove this helper.
static void append_bytes(string &str, const void *data, size_t len) {
  size_t old_size = str.size();
  str.resize(old_size + len);
  LIBC_NAMESPACE::inline_memcpy(str.data() + old_size, data, len);
}

namespace {

struct FakeNetworkSyscallPolicyData {
  FixedVector<tuple<int, int, int>, 10> socket_calls;
  FixedVector<ErrorOr<int>, 10> socket_results;
  FixedVector<tuple<int, size_t, int>, 10> sendto_calls;
  FixedVector<ErrorOr<ssize_t>, 10> sendto_results;
  // TODO: Move this into sendto_calls when FixedVector and tuple support
  // non-trivial types.
  string sendto_data;
  FixedVector<int, 10> close_calls;
  FixedVector<ErrorOr<int>, 10> close_results;
  FixedVector<tuple<int, size_t, int>, 10> recv_calls;
  // TODO: Make this ErrorOr<string> when ErrorOr and FixedVector support
  // non-trivial types.
  FixedVector<ErrorOr<span<const uint8_t>>, 10> recv_results;
};

template <FakeNetworkSyscallPolicyData *DATA> struct FakeNetworkSyscallPolicy {
  static ErrorOr<int> socket(int domain, int type, int protocol) {
    DATA->socket_calls.push_back(tuple<int, int, int>(domain, type, protocol));
    return pop_front_or(DATA->socket_results, Error(ENFILE));
  }

  static ErrorOr<ssize_t> sendto(int fd, const void *buf, size_t len, int flags,
                                 const struct sockaddr *, socklen_t) {
    DATA->sendto_calls.push_back(tuple<int, size_t, int>(fd, len, flags));
    append_bytes(DATA->sendto_data, buf, len);
    return pop_front_or(DATA->sendto_results, static_cast<ssize_t>(len));
  }

  static ErrorOr<ssize_t> recvfrom(int fd, void *buf, size_t len, int flags,
                                   struct sockaddr *, socklen_t *) {
    DATA->recv_calls.push_back(tuple<int, size_t, int>(fd, len, flags));
    ErrorOr<span<const uint8_t>> chunk =
        pop_front_or(DATA->recv_results, span<const uint8_t>());
    if (!chunk.has_value())
      return Error(chunk.error());
    if (chunk->size() > len)
      return Error(EINVAL);
    LIBC_NAMESPACE::inline_memcpy(buf, chunk->data(), chunk->size());
    return static_cast<ssize_t>(chunk->size());
  }

  static ErrorOr<int> close(int fd) {
    DATA->close_calls.push_back(fd);
    return pop_front_or(DATA->close_results, 0);
  }
};

struct LlvmLibcIfNameIndexTest
    : public LIBC_NAMESPACE::testing::ErrnoCheckingTest {
  static constexpr int FAKE_SOCKET = 47;
  static FakeNetworkSyscallPolicyData policy_data;
  using Policy = FakeNetworkSyscallPolicy<&policy_data>;

  void SetUp() override {
    ErrnoCheckingTest::SetUp();
    policy_data = {};
  }
  void validate_dump_request() {
    ASSERT_EQ(policy_data.sendto_calls.size(), size_t(1));
    ASSERT_EQ(get<0>(policy_data.sendto_calls[0]), FAKE_SOCKET);
    ASSERT_EQ(get<2>(policy_data.sendto_calls[0]), 0);
    ASSERT_EQ(policy_data.sendto_data.size(),
              static_cast<size_t>(NLMSG_LENGTH(sizeof(ifinfomsg))));
    auto *nlh =
        reinterpret_cast<struct nlmsghdr *>(policy_data.sendto_data.data());
    ASSERT_EQ(nlh->nlmsg_len,
              static_cast<uint32_t>(NLMSG_LENGTH(sizeof(ifinfomsg))));
    ASSERT_EQ(nlh->nlmsg_type, static_cast<uint16_t>(RTM_GETLINK));
    ASSERT_EQ(nlh->nlmsg_flags,
              static_cast<uint16_t>(NLM_F_REQUEST | NLM_F_DUMP));
    auto *ifm = reinterpret_cast<struct ifinfomsg *>(NLMSG_DATA(nlh));
    ASSERT_EQ(ifm->ifi_family, static_cast<unsigned char>(AF_UNSPEC));
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

/// A helper struct used to construct netlink messages. Name is the attribute
/// we're most interested in.
struct AttrName {
  string_view name;
  bool null_terminate = true;
  size_t payload_len() const { return name.size() + (null_terminate ? 1 : 0); }
  void write(uint8_t *dest) const {
    LIBC_NAMESPACE::inline_memcpy(dest, name.data(), name.size());
    if (null_terminate)
      dest[name.size()] = '\0';
  }
  static constexpr uint16_t TYPE = IFLA_IFNAME;
};

/// A helper struct used to construct netlink messages with integer attributes.
/// The attributes themselves are not important. We're just testing that we can
/// skip over them correctly.
template <uint16_t ATTR_TYPE> struct AttrInt {
  unsigned int value;
  size_t payload_len() const { return sizeof(unsigned int); }
  void write(uint8_t *dest) const {
    LIBC_NAMESPACE::inline_memcpy(dest, &value, sizeof(unsigned int));
  }
  static constexpr uint16_t TYPE = ATTR_TYPE;
};
using AttrMtu = AttrInt<IFLA_MTU>;
using AttrTxqlen = AttrInt<IFLA_TXQLEN>;

} // namespace

FakeNetworkSyscallPolicyData LlvmLibcIfNameIndexTest::policy_data;

template <typename... Attrs>
static size_t build_ifinfomsg_packet(uint8_t *buf, unsigned int index,
                                     const Attrs &...attrs) {
  size_t total_len = NLMSG_LENGTH(sizeof(struct ifinfomsg)) +
                     (0 + ... + RTA_ALIGN(RTA_LENGTH(attrs.payload_len())));

  LIBC_NAMESPACE::inline_memset(buf, 0, total_len);

  struct nlmsghdr *nh = reinterpret_cast<struct nlmsghdr *>(buf);
  nh->nlmsg_len = static_cast<uint32_t>(total_len);
  nh->nlmsg_type = RTM_NEWLINK;

  struct ifinfomsg *ifm = reinterpret_cast<struct ifinfomsg *>(NLMSG_DATA(nh));
  ifm->ifi_family = AF_UNSPEC;
  ifm->ifi_index = static_cast<int>(index);

  uint8_t *attr_ptr = reinterpret_cast<uint8_t *>(IFLA_RTA(ifm));

  auto write_attr = [&attr_ptr](const auto &attr) {
    size_t rta_len = RTA_LENGTH(attr.payload_len());
    struct rtattr *rta = reinterpret_cast<struct rtattr *>(attr_ptr);
    rta->rta_len = static_cast<unsigned short>(rta_len);
    rta->rta_type = attr.TYPE;
    attr.write(reinterpret_cast<uint8_t *>(RTA_DATA(rta)));
    attr_ptr += RTA_ALIGN(rta_len);
  };

  if constexpr (sizeof...(Attrs) > 0)
    (..., write_attr(attrs));

  return total_len;
}

static size_t build_nlmsg_done_packet(uint8_t *buf) {
  size_t total_len = NLMSG_LENGTH(sizeof(int));
  LIBC_NAMESPACE::inline_memset(buf, 0, total_len);
  struct nlmsghdr *nh = reinterpret_cast<struct nlmsghdr *>(buf);
  nh->nlmsg_len = static_cast<uint32_t>(total_len);
  nh->nlmsg_type = NLMSG_DONE;
  return total_len;
}

static size_t build_nlmsg_error_packet(uint8_t *buf, int errcode) {
  size_t total_len = NLMSG_LENGTH(sizeof(struct nlmsgerr));
  LIBC_NAMESPACE::inline_memset(buf, 0, total_len);
  struct nlmsghdr *nh = reinterpret_cast<struct nlmsghdr *>(buf);
  nh->nlmsg_len = static_cast<uint32_t>(total_len);
  nh->nlmsg_type = NLMSG_ERROR;
  struct nlmsgerr *err = reinterpret_cast<struct nlmsgerr *>(NLMSG_DATA(nh));
  err->error = errcode;
  return total_len;
}

static size_t build_generic_nlmsg_packet(uint8_t *buf, uint16_t type) {
  size_t total_len = NLMSG_LENGTH(0);
  LIBC_NAMESPACE::inline_memset(buf, 0, total_len);
  struct nlmsghdr *nh = reinterpret_cast<struct nlmsghdr *>(buf);
  nh->nlmsg_len = static_cast<uint32_t>(total_len);
  nh->nlmsg_type = type;
  return total_len;
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

TEST_F(LlvmLibcIfNameIndexSocketTest, RecvFailure) {
  policy_data.recv_results.push_back(Error(ETIMEDOUT));

  auto res = LIBC_NAMESPACE::net::if_nameindex<Policy>();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), ETIMEDOUT);
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
  ASSERT_TRUE(res.has_value());
  struct if_nameindex *list = res.value();
  ASSERT_NE(list, static_cast<struct if_nameindex *>(nullptr));
  ASSERT_EQ(list[0].if_index, 0u);
  ASSERT_EQ(list[0].if_name, static_cast<char *>(nullptr));

  validate_dump_request();
  LIBC_NAMESPACE::if_freenameindex(list);
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
