//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads, no-exceptions

// ASan seems to try to create threadsm which obviously doesn't work in this test.
// UNSUPPORTED: asan, hwasan

// UNSUPPORTED: c++03

// There is no way to limit the number of threads on windows
// UNSUPPORTED: windows

// macOS and FreeBSD seem to limit the number of processes, not threads via RLIMIT_NPROC.
// XFAIL: target={{.+}}-apple-{{.*}}
// XFAIL: freebsd

// z/OS does not have mechanism to limit the number of threads
// XFAIL: target={{.+}}-zos{{.*}}

// This test makes sure that we fail gracefully in care the thread creation fails. This is only reliably possible on
// systems that allow limiting the number of threads that can be created. See https://llvm.org/PR125428 for more details

#include <cassert>
#include <future>
#include <system_error>

#if __has_include(<sys/resource.h>)
#  include <sys/resource.h>
#  if defined(_AIX) && defined(RLIMIT_THREADS)
#    define TEST_RLIMIT RLIMIT_THREADS
#  elif defined(RLIMIT_NPROC)
#    define TEST_RLIMIT RLIMIT_NPROC
#  else
#    error "No known way to force only one thread being available"
#  endif
#else
#  error "No known way to force only one thread being available"
#endif
void force_thread_creation_failure() {
  rlimit lim = {1, 1};
  assert(setrlimit(TEST_RLIMIT, &lim) == 0);
}

int main(int, char**) {
  force_thread_creation_failure();

  try {
    std::future<int> fut = std::async(std::launch::async, [] { return 1; });
    assert(false);
  } catch (const std::system_error&) {
  }

  try {
    std::future<void> fut = std::async(std::launch::async, [] { return; });
    assert(false);
  } catch (const std::system_error&) {
  }

  return 0;
}
