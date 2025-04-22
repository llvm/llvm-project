// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_DETAIL_ENTRY
#define _LIBCPP_STACKTRACE_DETAIL_ENTRY

#include <__config>
#include <cstdint>
#include <string>

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __stacktrace {

/** Contains fields which will be used to generate the final `std::stacktrace_entry`.
This is an intermediate object which owns strings allocated via the caller-provided allocator,
which are later freed back to that allocator and converted to plain `std::string`s. */
struct _LIBCPP_HIDE_FROM_ABI entry {
  /** Caller's / faulting insn's address, including ASLR/slide */
  uintptr_t __addr_{};

  /** the address minus its image's slide offset */
  uintptr_t __addr_unslid_{};

  /** entry's description (symbol name) */
  std::pmr::string __desc_{};

  /** source file name */
  std::pmr::string __file_{};

  /** line number in source file */
  uint32_t __line_{};
};

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_DETAIL_ENTRY
