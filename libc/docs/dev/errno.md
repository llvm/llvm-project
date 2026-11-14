(errno)=

# `errno` and `libc_errno` in LLVM-libc

Many C and POSIX functions report errors by setting `errno`. Because LLVM-libc
supports multiple build and deployment modes ({doc}`../full_host_build`,
{doc}`../full_cross_build`, {doc}`../overlay_mode`, baremetal, GPU, and
header-only `libc/shared/`), internal code and unit tests access `errno`
through `libc_errno`.

## Targets Overview

`errno` support is split across three targets:

| CMake Target | Bazel Target | Header / Source | Purpose |
| :--- | :--- | :--- | :--- |
| `libc.hdr.errno_macros` | `:hdr_errno_macros` | `libc/hdr/errno_macros.h` | Error constants (`EINVAL`, `ERANGE`, `EDOM`, `ENOMEM`, etc.) without `libc_errno`. |
| `libc.src.__support.libc_errno` | `:__support_libc_errno` | `libc/src/__support/libc_errno.h` | Declares `libc_errno` (either as `#define libc_errno errno` or as `LIBC_INLINE_VAR constexpr Errno libc_errno{}`). |
| `libc.src.errno.errno` | `:errno` | `libc/src/errno/libc_errno.cpp` | Defines the backing storage (`thread_errno` or `shared_errno`), `extern "C" int *__llvm_libc_errno() noexcept`, and (when `LIBC_COPT_PROVIDE_GNU_ALIASES` is enabled) the `__errno_location` alias. |

## Usage Guidelines

### 1. Header Inclusion

- **Never `#include <errno.h>`** in `libc/src/**` or unit tests (`libc/test/src/**`).
- **Only need error constants (`EINVAL`, `ERANGE`, `EDOM`, etc.)?**
  Include `"hdr/errno_macros.h"`:

  ```c++
  #include "hdr/errno_macros.h"
  ```

- **Read or write `errno`?**
  Include `"src/__support/libc_errno.h"` (which also includes
  `"hdr/errno_macros.h"`) and use `libc_errno`:

  ```c++
  #include "src/__support/libc_errno.h"

  libc_errno = EINVAL;
  ```

### 2. Where to Set `libc_errno`

- Set `libc_errno` only at the public entrypoint boundary (`LLVM_LIBC_FUNCTION`)
  just before returning an error, or via `fputil::set_errno_if_required` in math
  implementations.
- Internal helpers in `libc/src/__support/` should not set `libc_errno`
  directly; return `ErrorOr<T>` or `cpp::optional<T>` instead and set
  `libc_errno` in the entrypoint:

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

### 3. Build Dependencies

| Target Type | Only Needs Error Constants (`EINVAL`, etc.) | Reads or Writes `libc_errno` |
| :--- | :--- | :--- |
| **`libc/src/__support/**`** | `libc.hdr.errno_macros` | `libc.src.__support.libc_errno` (never `libc.src.errno.errno`) |
| **Public Entrypoints (`libc/src/**`)** | `libc.hdr.errno_macros` | `libc.src.__support.libc_errno` (and `libc.src.errno.errno` for non-math `libc.a` entrypoints) |
| **Tests / Fuzzers / Benchmarks** | `libc.hdr.errno_macros` | Linked automatically (`libc.src.__support.libc_errno` and `libc.src.errno.errno`) |

- While public entrypoints normally cannot depend on other entrypoints,
  `libc.src.errno.errno` is explicitly allowed in `ALLOWED_DEPS` in
  `libc/cmake/modules/LLVMLibCObjectRules.cmake`.

### 4. Testing `errno`

- Test rules (`add_libc_test`, `add_fp_unittest`, `add_integration_test`,
  `add_libc_fuzzer`, and `perf_test`) automatically link `libc.src.errno.errno`.
- Unit tests that check `errno` should use `ErrnoCheckingTest`
  (`test/UnitTest/ErrnoCheckingTest.h`) or `ErrnoSetterMatcher`
  (`test/UnitTest/ErrnoSetterMatcher.h`):
  - `ErrnoCheckingTest` resets `libc_errno = 0` before each test and verifies
    `libc_errno == 0` at the end.
  - `ErrnoSetterMatcher` (`ASSERT_THAT(..., Fails(EINVAL))` / `Succeeds()`)
    checks and resets `libc_errno` automatically.
- **Header-only `libc/shared/*` tests (`libc/test/shared/*`)**:
  Compiled with `LIBC_ERRNO_MODE_SYSTEM_INLINE` in overlay mode; check `errno`
  directly when testing error conditions.

## `LIBC_ERRNO_MODE` Configurations

`LIBC_CONF_ERRNO_MODE` (passed to the compiler as `-DLIBC_ERRNO_MODE=<mode>`)
controls how `libc_errno` is implemented:

| Mode | `libc_errno` in `libc_errno.h` | Backing Storage in `libc_errno.cpp` | Typical Use Case |
| :--- | :--- | :--- | :--- |
| **`LIBC_ERRNO_MODE_DEFAULT`** | Resolves to `THREAD_LOCAL` or `SYSTEM_INLINE` (see below). | `thread_errno` | Default configuration. |
| **`LIBC_ERRNO_MODE_THREAD_LOCAL`** | Inline `Errno` wrapper calling `*__llvm_libc_errno()`. | `LIBC_THREAD_LOCAL int thread_errno;` | Full builds; overlay unit tests. |
| **`LIBC_ERRNO_MODE_SHARED`** | Inline `Errno` wrapper calling `*__llvm_libc_errno()`. | Global `int shared_errno;` | Single-threaded or thread-local-free targets (GPU, UEFI). |
| **`LIBC_ERRNO_MODE_EXTERNAL`** | Inline `Errno` wrapper calling `*__llvm_libc_errno()`. | Provided by external embedder. | Baremetal (`libc/config/baremetal/config.json`). |
| **`LIBC_ERRNO_MODE_UNDEFINED`** | Inline no-op (`operator=` is a no-op, `operator int()` returns `0`). | None (empty TU). | Minimal embedded environments without `errno`. |
| **`LIBC_ERRNO_MODE_SYSTEM_INLINE`** | `#define libc_errno errno` (from system `<errno.h>`). | None (empty TU). | Overlay public archive (`libllvmlibc.a`) and `libc/shared/*`. |

### How `LIBC_ERRNO_MODE_DEFAULT` Resolves

When `LIBC_ERRNO_MODE` is unset or set to `LIBC_ERRNO_MODE_DEFAULT`:

- **Full build (`LIBC_FULL_BUILD`)**: resolves to `LIBC_ERRNO_MODE_THREAD_LOCAL`.
- **Overlay public archive (`LIBC_COPT_PUBLIC_PACKAGING`)**: resolves to `LIBC_ERRNO_MODE_SYSTEM_INLINE` (`libc/shared/libc_common.h` and `libc/test/shared/` also configure `LIBC_ERRNO_MODE_SYSTEM_INLINE` in non-full builds).
- **Overlay tests, fuzzers, and benchmarks**: resolves to `LIBC_ERRNO_MODE_THREAD_LOCAL`.

## Why `libc_errno` Header and Runtime Storage Are Separated

Keeping `libc.src.__support.libc_errno` (header) separate from
`libc.src.errno.errno` (runtime object) is necessary for several reasons:

1. **Isolating unit tests from host `errno` in overlay mode**:
   Public overlay archives (`libllvmlibc.a`) are built with
   `-DLIBC_COPT_PUBLIC_PACKAGING` and use the host's `<errno.h>`
   (`LIBC_ERRNO_MODE_SYSTEM_INLINE`). Unit tests are built without
   `-DLIBC_COPT_PUBLIC_PACKAGING` and use LLVM-libc's internal `thread_errno`
   (`LIBC_ERRNO_MODE_THREAD_LOCAL`), preventing tested functions from clobbering
   the host `errno` used by the test harness. Conversely, if internal code or
   tests mistakenly `#include <errno.h>` and use `errno` directly, reads and
   writes will hit different thread-local variables in the same thread.
2. **Header-only `libc/shared/*` consumption**:
   External LLVM subprojects (such as `libc++` and `offload`) include headers
   under `libc/shared/`, which set `LIBC_ERRNO_MODE_SYSTEM_INLINE` and map
   `libc_errno` to `<errno.h>` without compiling or linking `libc_errno.cpp`.
3. **`errno`-free math and archive separation (`libc.a` vs. `libm.a` / `libmvec.a`)**:
   `FEnvImpl.h` includes `src/__support/libc_errno.h` to define
   `fputil::set_errno_if_required`, which is guarded by
   `#ifndef LIBC_MATH_HAS_NO_ERRNO`. Because `Errno` methods are inline,
   functions that never call `set_errno_if_required` (such as `ceil`, `floor`,
   `copysign`, vector math, or `LIBC_MATH_NO_ERRNO` builds) emit no reference to
   `__llvm_libc_errno()`, and `libc.src.errno.errno` stays in `libc.a` rather
   than being pulled into `libm.a` or `libmvec.a`.
4. **CMake directory ordering**:
   `libc/src/CMakeLists.txt` adds `src/__support` before `src/errno` because
   `libc.src.errno.errno` depends on `libc.src.__support.*`. Targets under
   `libc/src/__support/` therefore cannot depend on `libc.src.errno.errno`.
