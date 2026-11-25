(syscall_wrapper_refactor)=

# Syscall Wrappers

## Purpose

All Linux system calls in LLVM-libc are made through wrapper functions in one
directory: `src/__support/OSUtil/linux/syscall_wrappers/`. Code outside
`src/__support/OSUtil/` does not call `syscall_impl` or use `SYS_*` numbers
directly.

This provides several benefits:

- **Reusability**: Allows multiple entrypoints to share a single syscall
  implementation without public APIs depending on other public APIs.
- **Type Safety**: Using `ErrorOr<T>` ensures that error conditions are
  handled explicitly.
- **Consistency**: Standardizes the conversion of syscall return values into
  errno-compatible objects.
- **Maintainability**: Centralizes platform-specific syscall logic, making it
  easier to audit and update.

The migration of existing code to this structure is still in progress, so the
tree contains entrypoints that call `syscall_impl` themselves. Do not copy
them. New code uses a wrapper, and adds one if it does not exist yet.

## The Pattern

Each syscall has its own header-only library in the `syscall_wrappers`
directory. Wrappers live in the `linux_syscalls` namespace to make call sites
self-documenting and to clearly identify any leakage into OS-generic code.

### Example Wrapper (`src/__support/OSUtil/linux/syscall_wrappers/read.h`):

```c++
#include "hdr/types/ssize_t.h"
#include "src/__support/OSUtil/linux/syscall.h" // For syscall_checked
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<ssize_t> read(int fd, void *buf, size_t count) {
  return syscall_checked<ssize_t>(SYS_read, fd, buf, count);
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL
```

The entrypoint turns the result into the public contract. In outline:

```c++
LLVM_LIBC_FUNCTION(ssize_t, read, (int fd, void *buf, size_t count)) {
  ErrorOr<ssize_t> result = linux_syscalls::read(fd, buf, count);
  if (!result) {
    libc_errno = result.error();
    return -1;
  }
  return *result;
}
```

### Rules for wrappers

- **A wrapper represents an operation of the operating system**, not a POSIX
  function. Linux-specific system calls get wrappers like any other.
- **A wrapper hides which syscall number implements the operation.** Some
  architectures lack the older syscalls (there is no `SYS_mkdir` on AArch64,
  for instance), so the `mkdir` wrapper calls `SYS_mkdirat`. Callers do not
  repeat that choice.
- **Use `syscall_checked`.** It converts the kernel's error convention into an
  `ErrorOr<T>`. Do not call `syscall_impl` and check the sign of the result by
  hand.
- **Syscalls that cannot fail return the plain type.** `getpid` returns
  `pid_t`, not `ErrorOr<pid_t>`, and is one of the few places that use
  `syscall_impl`.
- **A wrapper never sets `errno`.** The entrypoint does. See
  {ref}`setting_errno`.

Other operating systems follow the same structure under their own directory
and namespace, for example `src/__support/OSUtil/freebsd/syscall_wrappers/`.

## How to Migrate

1. **Create the Wrapper**: Add a new header file in
   `src/__support/OSUtil/linux/syscall_wrappers/`.

2. **Update CMake**: Add a `add_header_library` target for the new wrapper in
   `src/__support/OSUtil/linux/syscall_wrappers/CMakeLists.txt`.

3. **Refactor Entrypoints**:

   - Include the new wrapper header (e.g., `read.h`).
   - Replace direct `syscall_impl` calls with
     `linux_syscalls::<function_name>`.
   - Update the entrypoint's `DEPENDS` in `CMakeLists.txt` to include the
     new wrapper target.

Migrate one system call per pull request.
