.. _syscall_wrapper_refactor:

==============================
Syscall Wrapper Refactoring
==============================

Purpose
=======

LLVM-libc is transitioning to a centralized system for Linux syscalls. The goal
is to move all direct ``syscall_impl`` calls into a dedicated directory:
``src/__support/OSUtil/linux/syscall_wrappers/``.

This refactor provides several benefits:

* **Reusability**: Allows multiple entrypoints to share a single syscall
  implementation without public APIs depending on other public APIs.
* **Type Safety**: Using ``ErrorOr<T>`` ensures that error conditions are
  handled explicitly.
* **Consistency**: Standardizes the conversion of syscall return values into
  errno-compatible objects.
* **Maintainability**: Centralizes platform-specific syscall logic, making it
  easier to audit and update.

The Pattern
===========

Each syscall should have its own header-only library in the ``syscall_wrappers``
directory. The wrapper function should return an ``ErrorOr<T>``. Wrappers live
in the ``linux_syscalls`` namespace to make call sites self-documenting and to
clearly identify any leakage into OS-generic code.

Example Wrapper (``src/__support/OSUtil/linux/syscall_wrappers/read.h``):
--------------------------------------------------------------------------

.. code-block:: c++

    #include "hdr/types/ssize_t.h"
    #include "src/__support/OSUtil/linux/syscall.h" // For syscall_impl
    #include "src/__support/common.h"
    #include "src/__support/error_or.h"
    #include "src/__support/macros/config.h"
    #include <sys/syscall.h> // For syscall numbers

    namespace LIBC_NAMESPACE_DECL {
    namespace linux_syscalls {

    LIBC_INLINE ErrorOr<ssize_t> read(int fd, void *buf, size_t count) {
      ssize_t ret = syscall_impl<ssize_t>(SYS_read, fd, buf, count);
      if (ret < 0) {
        return Error(-static_cast<int>(ret));
      }
      return ret;
    }

    } // namespace linux_syscalls
    } // namespace LIBC_NAMESPACE_DECL

How to Migrate
==============

1. **Cleanup Existing Implementation**: If the syscall was previously
   implemented in ``OSUtil/linux/fcntl.cpp`` (or similar), remove the old
   implementation to replace it with the new wrapper.
2. **Create the Wrapper**: Add a new header file in
   ``src/__support/OSUtil/linux/syscall_wrappers/``.
3. **Update CMake**: Add a ``add_header_library`` target for the new wrapper in
   ``src/__support/OSUtil/linux/syscall_wrappers/CMakeLists.txt``.
4. **Refactor Entrypoints**:

   * Include the new wrapper header (e.g., ``read.h``).
   * Replace direct ``syscall_impl`` calls with
     ``linux_syscalls::<function_name>``.
   * Update the entrypoint's ``DEPENDS`` in ``CMakeLists.txt`` to include the
     new wrapper target.
