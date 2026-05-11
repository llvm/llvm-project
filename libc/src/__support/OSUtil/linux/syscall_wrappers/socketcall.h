//===-- Implementation header for socketcall wrapper ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_SOCKETCALL_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_SOCKETCALL_H

#include "src/__support/CPP/type_traits/is_integral.h"
#include "src/__support/CPP/type_traits/is_pointer.h"
#include "src/__support/OSUtil/linux/syscall.h" // syscall_impl
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

#ifdef SYS_socketcall

template <typename T>
LIBC_INLINE unsigned long socketcall_arg_cast_impl(T val) {
  if constexpr (cpp::is_pointer_v<T>) {
    return reinterpret_cast<unsigned long>(val);
  } else {
    static_assert(cpp::is_integral_v<T>, "Expected integral or pointer type.");
    return static_cast<unsigned long>(val);
  }
}

template <typename ReturnType, typename... Args>
LIBC_INLINE ReturnType socketcall(int call, Args... args) {
  unsigned long sockcall_args[sizeof...(Args)] = {
      socketcall_arg_cast_impl(args)...};
  return LIBC_NAMESPACE::syscall_impl<ReturnType>(SYS_socketcall, call,
                                                  sockcall_args);
}

#endif // SYS_socketcall

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_SOCKETCALL_H
