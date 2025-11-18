//===-- Definition of macros from netinet/in.h ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_MACROS_NETINET_IN_MACROS_H
#define LLVM_LIBC_MACROS_NETINET_IN_MACROS_H

#include "../llvm-libc-types/in_addr_t.h"
#include "__llvm-libc-common.h"

#define IPPROTO_IP 0
#define IPPROTO_ICMP 1
#define IPPROTO_TCP 6
#define IPPROTO_UDP 17
#define IPPROTO_IPV6 41
#define IPPROTO_RAW 255

#define IPV6_UNICAST_HOPS 16
#define IPV6_MULTICAST_IF 17
#define IPV6_MULTICAST_HOPS 18
#define IPV6_MULTICAST_LOOP 19
#define IPV6_JOIN_GROUP 20
#define IPV6_LEAVE_GROUP 21
#define IPV6_V6ONLY 26

#define INADDR_ANY __LLVM_LIBC_CAST(static_cast, in_addr_t, 0x00000000)
#define INADDR_BROADCAST __LLVM_LIBC_CAST(static_cast, in_addr_t, 0xffffffff)
#define INADDR_NONE __LLVM_LIBC_CAST(static_cast, in_addr_t, 0xffffffff)

#define INET_ADDRSTRLEN 16
#define INET6_ADDRSTRLEN 46

#endif // LLVM_LIBC_MACROS_NETINET_IN_MACROS_H
