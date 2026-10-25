(errno)=

# `errno` and `libc_errno` in LLVM-libc

Many C standard library and POSIX functions report errors by setting `errno`.
Because LLVM-libc supports multiple deployment models—including
{doc}`../full_host_build`, {doc}`../full_cross_build`, {doc}`../overlay_mode`,
baremetal targets, GPUs, and header-only consumption via `libc/shared/`—access
to `errno` inside the library and its test suites is abstracted through
`libc_errno`.

This guide explains how `libc_errno` is architected, the supported
`LIBC_ERRNO_MODE` configurations, and the rules for using `libc_errno` in source
files, build targets, and tests.

## Architecture Overview: Header vs. Runtime Storage

In LLVM-libc, `errno` support is split across three distinct targets:

| CMake Target | Header / Source | Purpose |
| :--- | :--- | :--- |
| `libc.hdr.errno_macros` | `libc/hdr/errno_macros.h` | Provides standard error number constants (`EINVAL`, `ERANGE`, `EDOM`, `ENOMEM`, etc.) without declaring `libc_errno`. |
| `libc.src.__support.libc_errno` | `libc/src/__support/libc_errno.h` | Declares `libc_errno` (either as `#define libc_errno errno` or as `LIBC_INLINE_VAR constexpr Errno libc_errno{}`). |
| `libc.src.errno.errno` | `libc/src/errno/libc_errno.cpp` | Defines the runtime backing storage (`thread_errno` or `shared_errno`), the public C symbol `extern "C" int *__llvm_libc_errno() noexcept`, and (when `LIBC_COPT_PROVIDE_GNU_ALIASES` is enabled) its Linux/glibc ABI alias `extern "C" int *__errno_location() noexcept`. |

### Why `libc.src.__support.libc_errno` and `libc.src.errno.errno` Are Separated

Keeping the `libc_errno` header library separate from the `libc.src.errno.errno`
object target is required for five reasons:

1. **Isolating Unit Tests from Host `errno` in Overlay Mode ("Dual-TLS")**:
   - When building the public overlay archive (`libllvmlibc.a`), entrypoints are
     compiled with `-DLIBC_COPT_PUBLIC_PACKAGING`, which resolves `libc_errno`
     to the host system's `<errno.h>` (`LIBC_ERRNO_MODE_SYSTEM_INLINE`).
   - When building unit tests (`add_libc_test`), entrypoints and tests are
     compiled *without* `-DLIBC_COPT_PUBLIC_PACKAGING`, which resolves
     `libc_errno` to LLVM-libc's internal thread-local storage
     (`LIBC_ERRNO_MODE_THREAD_LOCAL`). This guarantees that running a test never
     clobbers the host libc's `errno` used by the test harness itself (and vice
     versa).

2. **Header-Only Shared Libraries (`libc/shared/*`)**:
   - External LLVM subprojects (such as `libc++` and the offloading runtime)
     consume internal headers under `libc/src/__support/` via `libc/shared/*.h`.
   - `libc/shared/libc_common.h` configures `LIBC_ERRNO_MODE_SYSTEM_INLINE` so
     that `libc_errno` maps directly to `<errno.h>` without compiling or linking
     `libc_errno.cpp`.

3. **`errno`-Free Math and Vector Math (`FEnvImpl.h`)**:
   - `libc/src/__support/FPUtil/FEnvImpl.h` includes `src/__support/libc_errno.h`
     to implement `fputil::set_errno_if_required(int err)`, which guards
     `libc_errno = err;` behind `if (!cpp::is_constant_evaluated())` and
     `#ifndef LIBC_MATH_HAS_NO_ERRNO`.
   - Vector math functions (`libc/src/__support/mathvec/`), builds with
     `LIBC_MATH_NO_ERRNO` or `LIBC_MATH_FAST`, and scalar math functions that
     only manipulate floating-point exceptions or rounding modes (`ceil`,
     `floor`, `trunc`, `round`, `rint`, `copysign`, `frexp`, `modf`, etc.)
     include `FEnvImpl.h` but emit no link-time reference to
     `__llvm_libc_errno()`.

4. **Archive Separation (`libc.a` vs. `libm.a` / `libmvec.a`)**:
   - In full builds, `libc.src.errno.errno` belongs in `libc.a`
     (`TARGET_LIBC_ENTRYPOINTS`) and must not be packaged into `libm.a` or
     `libmvec.a`.

5. **Topological Directory Ordering**:
   - `libc/src/CMakeLists.txt` configures `add_subdirectory(__support)` before
     `add_subdirectory(errno)` because `libc.src.errno.errno` depends on
     `libc.src.__support.*`. Targets under `libc/src/__support/` therefore
     cannot depend on `libc.src.errno.errno`.

## `LIBC_ERRNO_MODE` Configurations

The behavior of `libc_errno` is controlled by the `LIBC_CONF_ERRNO_MODE` CMake
option (passed to the compiler as `-DLIBC_ERRNO_MODE=<mode>`). In
`libc/src/__support/libc_errno.h`, `Errno::operator=(int)` and
`Errno::operator int()` are defined inline (`LIBC_INLINE`), delegating to
`*__llvm_libc_errno()` when runtime storage is used:

| Mode | `libc_errno` Implementation | Backing Storage (`libc_errno.cpp`) | Typical Use Case |
| :--- | :--- | :--- | :--- |
| **`LIBC_ERRNO_MODE_DEFAULT`** | Auto-selects `SYSTEM_INLINE` or `THREAD_LOCAL` (see below). | `thread_errno` (`THREAD_LOCAL`) | Default CMake / Bazel configuration. |
| **`LIBC_ERRNO_MODE_THREAD_LOCAL`** | `Errno` inline wrapper calling `*__llvm_libc_errno()`. | `LIBC_THREAD_LOCAL int thread_errno;` | Full builds on Linux/macOS/Windows; overlay unit tests. |
| **`LIBC_ERRNO_MODE_SHARED`** | `Errno` inline wrapper calling `*__llvm_libc_errno()`. | Global `int shared_errno;` | Single-threaded or thread-local-free targets (GPU, UEFI). |
| **`LIBC_ERRNO_MODE_EXTERNAL`** | `Errno` inline wrapper calling `*__llvm_libc_errno()`. | None (provided by external embedder). | Baremetal targets (`libc/config/baremetal/config.json`). |
| **`LIBC_ERRNO_MODE_UNDEFINED`** | `Errno` inline no-op (`operator=` does nothing, `operator int()` returns `0`). | None (empty TU). | Minimal embedded environments without `errno`. |
| **`LIBC_ERRNO_MODE_SYSTEM_INLINE`** | `#define libc_errno errno` (from system `<errno.h>`). | None (empty TU). | Overlay public archive (`libllvmlibc.a`) and `libc/shared/*`. |

### How `LIBC_ERRNO_MODE_DEFAULT` Resolves

When `LIBC_ERRNO_MODE` is unset or set to `LIBC_ERRNO_MODE_DEFAULT`,
`libc/src/__support/libc_errno.h` resolves it as follows:

- If `LIBC_FULL_BUILD` is defined: resolves to `LIBC_ERRNO_MODE_THREAD_LOCAL`.
- Otherwise (overlay / non-full build):
  - If `LIBC_COPT_PUBLIC_PACKAGING` is defined (building `libllvmlibc.a`) **or**
    `LLVM_LIBC_SHARED_LIBC_COMMON_H` is defined (included after
    `libc/shared/libc_common.h`): resolves to `LIBC_ERRNO_MODE_SYSTEM_INLINE`.
  - Otherwise (building unit tests, fuzzers, or benchmarks): resolves to
    `LIBC_ERRNO_MODE_THREAD_LOCAL`.

## Usage Rules for Contributors

### 1. Header Inclusion in Source and Test Files

- **Never `#include <errno.h>`** in implementation (`libc/src/**`) or test
  (`libc/test/**`) files.
- **If you only need error constants (`EINVAL`, `ERANGE`, `EDOM`, `ENOMEM`, etc.)**:
  Include `"hdr/errno_macros.h"`:

  ```c++
  #include "hdr/errno_macros.h"
  ```

- **If you read or write `errno`**:
  Include `"src/__support/libc_errno.h"` and access `libc_errno`:

  ```c++
  #include "src/__support/libc_errno.h"

  // Setting errno:
  libc_errno = EINVAL;
  ```

  *(Note: `"src/__support/libc_errno.h"` transitively includes
  `"hdr/errno_macros.h"`, though including `"hdr/errno_macros.h"` explicitly when
  using error macros is good practice.)*

### 2. Where to Set `libc_errno`

- Set `libc_errno` **only** at the top-level public entrypoint boundary
  (`LLVM_LIBC_FUNCTION`) just before returning an error value, or via
  `fputil::set_errno_if_required` in math implementations.
- Internal helper functions in `libc/src/__support/` should **not** mutate
  `libc_errno` directly (unless specifically designed as an `errno`-reporting
  layer). Instead, return `ErrorOr<T>` (`cpp::expected<T, int>`) or
  `cpp::optional<T>` and convert the error at the entrypoint boundary:

  ```c++
  LLVM_LIBC_FUNCTION(int, close, (int fd)) {
    auto res = linux_syscalls::close(fd);
    if (!res) {
      libc_errno = res.error();
      return -1;
    }
    return 0;
  }
  ```

### 3. CMake `DEPENDS` Rules

Choose the CMake dependency that matches how the target uses `errno`:

1. **Internal Support Libraries (`libc/src/__support/**`)**:
   - Only uses error constants (`EINVAL`, etc.):
     `DEPENDS libc.hdr.errno_macros`
   - Reads or writes `libc_errno`:
     `DEPENDS libc.src.__support.libc_errno`
   - **Never** add `libc.src.errno.errno` to a target in `libc/src/__support/**`.

2. **Public Entrypoints (`add_entrypoint_object` under `libc/src/`)**:
   - Only uses error constants (for example, `pthread_*` functions that return
     `EINVAL` directly without setting `errno`):
     `DEPENDS libc.hdr.errno_macros`
   - Reads or writes `libc_errno`:
     `DEPENDS libc.src.__support.libc_errno` (and `libc.src.errno.errno` for
     non-math `libc.a` entrypoints). Note that `libc.src.errno.errno` is the
     only entrypoint target permitted in another entrypoint's `DEPENDS` list
     (`ALLOWED_DEPS` in `LLVMLibCObjectRules.cmake`).

3. **Header-Only Shared Tests (`libc/test/shared/*`)**:
   - Must include `shared/<header>.h` (which includes `shared/libc_common.h`)
     **before** any internal headers or test headers so that
     `LLVM_LIBC_SHARED_LIBC_COMMON_H` is defined prior to
     `src/__support/libc_errno.h`.
   - When checking `errno` in `libc/test/shared/*`, inspect `errno` directly
     (since `shared/` uses `LIBC_ERRNO_MODE_SYSTEM_INLINE` in overlay builds and
     public `<errno.h>` in full builds).

4. **Unit Tests, Integration Tests, Fuzzers, and Performance Tests**:
   - `add_libc_test`, `add_fp_unittest`, `add_integration_test`,
     `add_libc_fuzzer`, and `perf_test` automatically link
     `libc.src.errno.errno`.
   - Unit tests that verify `errno` behavior should use `ErrnoCheckingTest`
     (`test/UnitTest/ErrnoCheckingTest.h`) and/or `ErrnoSetterMatcher`
     (`test/UnitTest/ErrnoSetterMatcher.h`):
     - `ErrnoCheckingTest` resets `libc_errno = 0` before each `TEST_F` body and
       asserts that `libc_errno` is `0` at the end of the test (requiring tests
       that intentionally trigger errors to validate and clear `libc_errno`).
     - `ErrnoSetterMatcher` (`ASSERT_THAT(..., Fails(EINVAL))` / `Succeeds()`)
       automatically checks and resets `libc_errno`.
