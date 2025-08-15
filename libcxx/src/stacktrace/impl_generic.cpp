//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/*
"Generic" implementation for any platform that doesn't have its own special implementation.
(Currently this means any platform other than Windows)
*/

#if !defined(_WIN32)

#  include <__config>
#  include <__stacktrace/stacktrace_entry.h>

#  include "stacktrace/images.h"
#  include "stacktrace/tools/tools.h"
#  include "stacktrace/unwinding.h"

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE _LIBCPP_EXPORTED_FROM_ABI void
base::current_impl(size_t skip, size_t max_depth) {
  if (!max_depth) [[unlikely]] {
    return;
  }

  /*
  Build up the stacktrace entries and fill out their fields the best we can.
  An `entry_base` structure looks more or less like:

    __addr_     uintptr_t   Instruction address (of a call instruction, faulting insn, ...)
    __desc_     string      "Description" which we'll say is synonymous with "symbol"
    __file_     string      Source filename, or if that's not available, program/library path
    __line_     int         Line number within the source file, or failing that, zero
    __image_    image*      The image, loaded in by the OS, containing that address (program, library)

  On Windows, there are DLLs which take care of all this (dbghelp, psapi), with essentially
  zero overlap with any other OS, so it's in its own file (impl_windows).  Otherwise, i.e.
  on non-Windows platforms, taking a stacktrace looks like:

  0. Create the basic_stacktrace, its internal vector, using provided allocator.
     (This was handled by the `current` member functions inside `basic_stacktrace`.)
  1. Collect instruction addresses to build up the entries, observing skip and max_depth.
     The unwinding library we have available to us should take care of walking the call stack
     and finding addresses.
  2. Resolve addresses into the program images (executable, libraries); normalize the addresses
     from step 1, as they were seen in the wild, into program-image-relative addresses
     (i.e. deal with ASLR)
  3. Resolve adjusted addresses into their symbols; some environments provide this out of the box
     (MacOS) and others make this less easy (Linux)
  4. To get the source file and line number, we have to dig through debug information (DWARF format);
     we might need the help of an external tool or library.
     4A: Ideally we would have a library inside libcxx, possibly refactored from somewhere within
         compiler-rt, lldb, llvm-symbolizer, that could handle all of this.
         (XXX we don't have this currently)
     4B: If the local system happens to have a library that does this, that will work too.
         Look for: libbacktrace, libdwarf, etc.
         (XXX we don't do this yet)
     4C: Use an external tool (i.e. spawn a child process) which can do this.

  */

  // (1) Collect instruction addresses; build vector, populate their `__addr_`'s
  unwind_addrs(*this, skip + 1, max_depth);
  if (!__entries_size_()) {
    return;
  }

  // (2) Map these addresses to their respective program images, populate `__image_`
  find_images();

  // (3) Use system loader and/or `dl` to get symbols
  find_symbols();

  // (4C) Use an external tool to get source file/line, as well as any missing symbols
  find_source_locs();
}

void base::find_images() {
  images images;
  size_t i  = 0;
  auto* it  = entries_begin();
  auto* end = entries_end();
  while (it != end) {
    auto& entry = *it++;
    images.find(&i, entry.__addr_);
    if (auto& image = images[i]) {
      entry.__image_ = &image;
      // While we're in this loop, get the executable's path, and tentatively use this for source file.
      entry.assign_file(__strings_.make_str(image.name_));
    }
  }
}

void base::find_symbols() {}

void base::find_source_locs() {
#  if __has_include(<spawn.h>) && _LIBCPP_STACKTRACE_ALLOW_TOOLS_AT_RUNTIME
  (void)(false                                                                                  //
         || (__has_working_executable<atos>() && __run_tool<atos>(*this))                       //
         || (__has_working_executable<llvm_symbolizer>() && __run_tool<llvm_symbolizer>(*this)) //
         || (__has_working_executable<addr2line>() && __run_tool<addr2line>(*this)));           //
#  endif
}

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif
