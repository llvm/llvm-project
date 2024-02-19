//===-- Definition of epoll_data type -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_EPOLL_DATA_H__
#define __LLVM_LIBC_TYPES_EPOLL_DATA_H__

union epoll_data {
  void *ptr;
  int fd;
  __UINT32_TYPE__ u32;
  __UINT64_TYPE__ u64;
};

typedef union epoll_data epoll_data_t;

#endif // __LLVM_LIBC_TYPES_EPOLL_DATA_H__
