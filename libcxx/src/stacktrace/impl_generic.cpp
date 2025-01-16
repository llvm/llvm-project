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
     we might need the help of a library.
  */

  unwind_addrs(*this, skip + 1, max_depth);
  if (__entry_iters_().size()) {
    find_images();
    find_symbols();
    find_source_locs();
  }
}

void base::find_images() {
  images images;
  size_t i = 0;
  for (auto& entry : __entry_iters_()) {
    images.find(&i, entry.__addr_);
    if (auto& image = images[i]) {
      entry.__image_ = &image;
      // While we're in this loop, get the executable's path, and tentatively use this for source file.
      entry.assign_file(__create_str()).assign(image.name_);
    }
  }
}

void base::find_symbols() {
  // TODO
}

void base::find_source_locs() {
  // TODO
}

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif
