//===-- Implementation header of internal fcntl function ------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_FCNTL_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_FCNTL_H

namespace LIBC_NAMESPACE::internal {

int fcntl(int fd, int cmd, void *arg = nullptr);

} // namespace LIBC_NAMESPACE::internal

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_FCNTL_H
