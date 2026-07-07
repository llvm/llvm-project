//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Internal templatized implementation of if_nameindex for Linux.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_NET_LINUX_IF_NAMEINDEX_H
#define LLVM_LIBC_SRC_NET_LINUX_IF_NAMEINDEX_H

#include "hdr/errno_macros.h"
#include "hdr/net_if_macros.h"
#include "hdr/sys_socket_macros.h"
#include "hdr/types/socklen_t.h"
#include "hdr/types/ssize_t.h"
#include "hdr/types/struct_if_nameindex.h"
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/scope.h"
#include "src/__support/alloc-checker.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/string/memory_utils/inline_memset.h"
#include "src/string/string_utils.h"

#include <linux/netlink.h>
#include <linux/rtnetlink.h>

namespace LIBC_NAMESPACE_DECL {
namespace net {

template <typename Policy> ErrorOr<struct if_nameindex *> if_nameindex() {
  ErrorOr<int> fd_or_err =
      Policy::socket(AF_NETLINK, SOCK_RAW | SOCK_CLOEXEC, NETLINK_ROUTE);
  if (!fd_or_err.has_value())
    return Error(fd_or_err.error());
  int fd = *fd_or_err;
  cpp::scope_exit close_fd([fd]() { Policy::close(fd); });

  struct {
    struct nlmsghdr nlh;
    struct ifinfomsg ifm;
  } req = {};
  static_assert(sizeof(req) >= NLMSG_LENGTH(sizeof(req.ifm)));
  req.nlh.nlmsg_len = NLMSG_LENGTH(sizeof(req.ifm));
  req.nlh.nlmsg_type = RTM_GETLINK;
  req.nlh.nlmsg_flags = NLM_F_REQUEST | NLM_F_DUMP;
  req.ifm.ifi_family = AF_UNSPEC;

  ErrorOr<ssize_t> send_res =
      Policy::sendto(fd, &req, req.nlh.nlmsg_len, 0, nullptr, 0);
  if (!send_res.has_value())
    return Error(send_res.error());

  alignas(struct nlmsghdr) uint8_t buf[4096];
  ErrorOr<ssize_t> recv_res =
      Policy::recvfrom(fd, buf, sizeof(buf), 0, nullptr, nullptr);
  if (!recv_res.has_value())
    return Error(recv_res.error());

  close_fd.release();
  if (ErrorOr<int> close_res = Policy::close(fd); !close_res.has_value())
    return Error(close_res.error());

  // TODO: Read more than one message.
  // TODO: Read more than one interface per message.
  // TODO: Deduplicate interfaces to handle restarts.
  auto len = static_cast<size_t>(*recv_res);
  for (auto *nh = reinterpret_cast<struct nlmsghdr *>(buf); NLMSG_OK(nh, len);
       nh = NLMSG_NEXT(nh, len)) {
    if (nh->nlmsg_type == NLMSG_DONE)
      break;
    if (nh->nlmsg_type == NLMSG_ERROR) {
      if (nh->nlmsg_len < NLMSG_LENGTH(sizeof(struct nlmsgerr)))
        return Error(EINVAL);
      auto *err = reinterpret_cast<struct nlmsgerr *>(NLMSG_DATA(nh));
      if (err->error == 0) {
        // Zero means an ACK, which we shouldn't get because we didn't ask for
        // it...
        continue;
      }
      return Error(-err->error);
    }
    if (nh->nlmsg_type != RTM_NEWLINK)
      continue;
    if (nh->nlmsg_len < NLMSG_LENGTH(sizeof(struct ifinfomsg)))
      continue;

    auto *ifm = reinterpret_cast<struct ifinfomsg *>(NLMSG_DATA(nh));
    size_t attrlen = nh->nlmsg_len - NLMSG_LENGTH(sizeof(struct ifinfomsg));

    auto *rta = reinterpret_cast<struct rtattr *>(
        reinterpret_cast<uint8_t *>(ifm) +
        NLMSG_ALIGN(sizeof(struct ifinfomsg)));
    for (; RTA_OK(rta, attrlen); rta = RTA_NEXT(rta, attrlen)) {
      if (rta->rta_type == IFLA_IFNAME) {
        size_t rta_payload_len = RTA_PAYLOAD(rta);
        auto index = static_cast<unsigned int>(ifm->ifi_index);
        const char *name_data = reinterpret_cast<const char *>(RTA_DATA(rta));
        size_t name_len = internal::strnlen(name_data, rta_payload_len);

        size_t total_size = 2 * sizeof(struct if_nameindex) + name_len + 1;
        AllocChecker ac;
        uint8_t *buffer = new (ac) uint8_t[total_size];
        if (!ac)
          return Error(ENOMEM);

        auto *result = reinterpret_cast<struct if_nameindex *>(buffer);
        char *string_ptr = reinterpret_cast<char *>(result + 2);

        result[0].if_index = index;
        result[0].if_name = string_ptr;
        inline_memcpy(string_ptr, name_data, name_len);
        string_ptr[name_len] = '\0';

        result[1].if_index = 0;
        result[1].if_name = nullptr;

        return result;
      }
    }
  }

  AllocChecker ac;
  uint8_t *buffer = new (ac) uint8_t[sizeof(struct if_nameindex)];
  if (!ac)
    return Error(ENOMEM);
  auto *result = reinterpret_cast<struct if_nameindex *>(buffer);
  result[0] = {};
  return result;
}

} // namespace net
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_NET_LINUX_IF_NAMEINDEX_H
