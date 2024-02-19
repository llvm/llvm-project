//===-- Definition of epoll_event type ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_EPOLL_EVENT_H__
#define __LLVM_LIBC_TYPES_EPOLL_EVENT_H__

#include <llvm-libc-types/struct_epoll_data.h>

typedef struct epoll_event {
  __UINT32_TYPE__ events;
  epoll_data_t data;
} epoll_event;

#endif // __LLVM_LIBC_TYPES_EPOLL_EVENT_H__
