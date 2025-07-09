//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_LINUX_IMPL
#define _LIBCPP_STACKTRACE_LINUX_IMPL

#include <__stacktrace/base.h>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

struct linux {
  builder& builder_;

#if defined(__linux__)
  // defined in linux.cpp
  void ident_modules();
  void symbolize();

private:
  void resolve_main_elf_syms(std::string_view elf_name);
#else
  // inline-able dummy definitions
  void ident_modules() {}
  void symbolize() {}
#endif
};

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_LINUX_IMPL
