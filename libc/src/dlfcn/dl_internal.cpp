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

#include "src/dlfcn/dl_internal.h"

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace dl {

// Recursive, because the initialisers of what was just loaded run while it
// is held, and an initialiser is entitled to call back into the loader:
// dlopen for something it needs, or dlsym for something it was given. A
// plain mutex leaves the one thread waiting on itself, which is what
// systemd did as PID 1, with nothing left to wake it. glibc's load lock is
// recursive for the same reason.
Mutex dl_mutex(false, /*is_recursive=*/true, false, false);
Mutex dl_write_lock(false, /*is_recursive=*/true, false, false);
const char *dl_last_error = nullptr;

namespace {
char dl_error_text[4352];
} // anonymous namespace

void set_error(const char *subject, const char *message) {
  size_t used = 0;
  auto append = [&used](const char *text) {
    for (; *text != '\0' && used + 1 < sizeof(dl_error_text); ++text)
      dl_error_text[used++] = *text;
  };
  append(subject);
  append(": ");
  append(message);
  dl_error_text[used] = '\0';
  dl_last_error = dl_error_text;
}

} // namespace dl
} // namespace LIBC_NAMESPACE_DECL
