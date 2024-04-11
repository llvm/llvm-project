//===-- Implementation header for epoll_create function ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_EPOLL_EPOLL_CREATE_H
#define LLVM_LIBC_SRC_SYS_EPOLL_EPOLL_CREATE_H

namespace LIBC_NAMESPACE {

int epoll_create(int size);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_SYS_EPOLL_EPOLL_CREATE_H
