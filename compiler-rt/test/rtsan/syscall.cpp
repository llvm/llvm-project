// RUN: %clangxx -fsanitize=realtime %s -o %t
// RUN: %env_rtsan_opts="halt_on_error=false" %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK-RTSAN,CHECK

// RUN: %clangxx %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// UNSUPPORTED: ios

// Intent: Ensure the `syscall` call behaves in the same way with/without the
//         sanitizer disabled

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

const char *GetTemporaryFilePath() { return "/tmp/rtsan_syscall_test.txt"; }

void custom_assert(bool condition, const char *message) {
  if (!condition) {
    fprintf(stderr, "ASSERTION FAILED: %s\n", message);
    exit(1);
  }
}

class ScopedFileCleanup {
public:
  [[nodiscard]] ScopedFileCleanup() = default;
  ~ScopedFileCleanup() {
    if (access(GetTemporaryFilePath(), F_OK) != -1)
      unlink(GetTemporaryFilePath());
  }
};

// Apple has deprecated `syscall`, ignore that error
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
int main() [[clang::nonblocking]] {
  ScopedFileCleanup cleanup;

  {
    int fd = syscall(SYS_openat, AT_FDCWD, GetTemporaryFilePath(),
                     O_CREAT | O_WRONLY, 0644);

    custom_assert(fd != -1, "Failed to open file - write");

    int written = syscall(SYS_write, fd, "Hello, world!", 13);
    custom_assert(written == 13, "Failed to write to file");

    custom_assert(syscall(SYS_close, fd) == 0, "Failed to close file - write");
  }

  {
    int fd = syscall(SYS_openat, AT_FDCWD, GetTemporaryFilePath(), O_RDONLY);
    custom_assert(fd != -1, "Failed to open file - read");

    char buffer[13];
    int read = syscall(SYS_read, fd, buffer, 13);
    custom_assert(read == 13, "Failed to read from file");

    custom_assert(memcmp(buffer, "Hello, world!", 13) == 0,
                  "Read data does not match written data");

    custom_assert(syscall(SYS_close, fd) == 0, "Failed to close file - read");
  }

  unlink(GetTemporaryFilePath());
  printf("DONE\n");
}
#pragma clang diagnostic pop

// CHECK-NOT: ASSERTION FAILED
// CHECK-RTSAN-COUNT-6: Intercepted call to real-time unsafe function `syscall`

// CHECK: DONE
