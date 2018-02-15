//===-- netbsd_syscall_hooks.h --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of public sanitizer interface.
//
// System call handlers.
//
// Interface methods declared in this header implement pre- and post- syscall
// actions for the active sanitizer.
// Usage:
//   __sanitizer_syscall_pre_getfoo(...args...);
//   long long res = syscall(SYS_getfoo, ...args...);
//   __sanitizer_syscall_post_getfoo(res, ...args...);
//
// DO NOT EDIT! THIS FILE HAS BEEN GENERATED!
//
// Generated with: generate_netbsd_syscalls.awk
// Generated date: 2018-02-15
// Generated from: syscalls.master,v 1.291 2018/01/06 16:41:23 kamil Exp
//
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_NETBSD_SYSCALL_HOOKS_H
#define SANITIZER_NETBSD_SYSCALL_HOOKS_H

#ifdef __cplusplus
extern "C" {
#endif

// Private declarations. Do not call directly from user code. Use macros above.

// DO NOT EDIT! THIS FILE HAS BEEN GENERATED!

#ifdef __cplusplus
} // extern "C"
#endif

// DO NOT EDIT! THIS FILE HAS BEEN GENERATED!

#endif // SANITIZER_NETBSD_SYSCALL_HOOKS_H
