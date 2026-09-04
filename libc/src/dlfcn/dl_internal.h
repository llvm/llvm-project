//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Shared state of the dlfcn functions
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_DLFCN_DL_INTERNAL_H
#define LLVM_LIBC_SRC_DLFCN_DL_INTERNAL_H

#include "hdr/types/size_t.h"
#include "src/__support/elf/passive_abi.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/mutex.h"

namespace LIBC_NAMESPACE_DECL {
namespace dl {

// A handle is the module's index plus one, so that it is never null and
// carries no pointer the caller could dereference.
LIBC_INLINE void *handle_for(size_t index) {
  return reinterpret_cast<void *>(index + 1);
}

// Returns false when the handle did not come from dlopen.
LIBC_INLINE bool index_for(void *handle, size_t &index) {
  const size_t value = reinterpret_cast<size_t>(handle);
  if (value == 0 || value > elf::loaded_modules().count)
    return false;
  index = value - 1;
  return true;
}

extern Mutex dl_mutex;

// Held while the module set is being changed, and by anything that reads it
// without holding dl_mutex. The load lock above is held across the
// initialisers of whatever was opened, which can wait on another thread, so
// a thread that is only walking the modules, an unwinder in
// dl_iterate_phdr, must not wait on that one.
extern Mutex dl_write_lock;

// The message dlerror will report, or null when there is nothing to report.
// Either a string literal or the text in dl_error_text.
extern const char *dl_last_error;

LIBC_INLINE void set_error(const char *message) { dl_last_error = message; }

// A message that names what it is about, as glibc's do: "libfoo.so: cannot
// open shared object file". The parts come from the caller, so they are
// copied, and cut short if they do not fit.
void set_error(const char *subject, const char *message);

} // namespace dl
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_DLFCN_DL_INTERNAL_H
