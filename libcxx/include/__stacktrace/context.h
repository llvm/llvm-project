// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_CONTEXT
#define _LIBCPP_STACKTRACE_CONTEXT

#include <__config>
#include <__memory_resource/memory_resource.h>
#include <__memory_resource/polymorphic_allocator.h>
#include <__vector/pmr.h>
#include <__vector/vector.h>
#include <cstddef>
#include <string>

#include <__stacktrace/alloc.h>
#include <__stacktrace/entry.h>

_LIBCPP_BEGIN_NAMESPACE_STD

struct _LIBCPP_HIDDEN alloc;
struct _LIBCPP_HIDDEN entry;

namespace __stacktrace {

/** Represents the state of the current in-progress stacktrace operation.  This includes
the working list of `entry` objects, as well as the caller-provided allocator (wrapped
in `polymorphic_allocator`) and a handful of utility functions involving that allocator. */
struct _LIBCPP_HIDDEN context {
  /** Encapsulates and type-removes the caller's allocator. */
  alloc& __alloc_;

  /** The working set of entry data; these will later be copied to their final `stacktrace_entry` objects. */
  std::pmr::list<entry> __entries_;

  /** Path to this process's main executable. */
  std::pmr::string __main_prog_path_;

  _LIBCPP_HIDDEN explicit context(alloc& __byte_alloc)
      : __alloc_(__byte_alloc), __entries_(&__alloc_), __main_prog_path_(__alloc_.new_string()) {}

  _LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE _LIBCPP_HIDDEN void do_stacktrace(size_t __skip, size_t __max_depth);
};

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_CONTEXT
