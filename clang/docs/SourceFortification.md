# Source Fortification

## Introduction

Clang provides built-in compile-time source fortification diagnostics under
{ref}`-Wfortify-source` (enabled by default) to catch buffer overflows, format
string truncations, and invalid arguments in calls to supported C and POSIX
library functions.

`-Wfortify-source` diagnoses calls when buffer sizes and access lengths (or
argument values) can be constant-evaluated. It does not flag cases where
proving an overflow requires optimizations (such as inlining, constant
propagation, or value-range analysis).

## Supported Checks

`-Wfortify-source` can diagnose:

1. **Destination buffer overflows and format truncation**: When a
   constant-evaluated size argument, source string literal length, or minimum
   output length of a string literal format string exceeds the known size of
   the destination buffer (or, for bounded format functions, exceeds the
   constant-evaluated size bound):
   - `<string.h>` / `<strings.h>`: `memcpy`, `memmove`, `memset`, `mempcpy`,
     `bcopy`, `bzero`, `strcpy`, `stpcpy`, `strcat`, `strncpy`, `stpncpy`,
     `strncat`, `strlcpy`, `strlcat` (and their `__builtin_` variants).
   - `<stdio.h>`: `sprintf`, `vsprintf`, `snprintf`, `vsnprintf` (format
     overflow and truncation are also controlled by {ref}`-Wformat-overflow` and
     {ref}`-Wformat-truncation`), `scanf`, `fscanf`, `sscanf`, `fgets`, `fread`.
   - `<poll.h>`: `poll`, `ppoll`, `ppoll64`.
   - `<sys/socket.h>`: `recv`, `recvfrom`.

2. **Source buffer overreads**: When a constant-evaluated byte count exceeds
   the known size of the source buffer:
   - `<stdio.h>`: `fwrite`.

3. **Invalid constant arguments**:
   - `<sys/stat.h>`: `umask` when called with constant mode bits outside `0777`
     that are silently ignored.

### Related Diagnostic Flags

Several closely related compile-time bounds checks are controlled by separate
diagnostic groups:

- {ref}`-Wformat-overflow` and {ref}`-Wformat-truncation`: Enabled as subgroups
  of `-Wfortify-source`; control format-string destination overflow and
  truncation warnings for `sprintf`, `vsprintf`, `snprintf`, and `vsnprintf`.
- {ref}`-Wbuiltin-memcpy-chk-size`: Diagnoses when a constant-evaluated byte
  count passed to most `__builtin___*_chk` functions exceeds the
  constant-evaluated destination object size.
- {ref}`-Wstringop-overread`: Diagnoses when a constant-evaluated byte count
  passed to memory functions (such as `memcpy`, `memmove`, `mempcpy`, `bcopy`,
  `memchr`, `memcmp`, and `bcmp`) exceeds the known size of the source buffer.

## Differences from `_FORTIFY_SOURCE`

The [`_FORTIFY_SOURCE`](https://sourceware.org/glibc/manual/latest/html_node/Source-Fortification.html)
macro in C standard libraries (such as glibc and Android's Bionic) hardens
calls to C and POSIX functions by rewriting them in system headers into
fortified wrappers and `__builtin___*_chk` intrinsics (typically when
optimization is enabled), combining compile-time diagnostics with runtime
bounds checks. Clang's `-Wfortify-source` flag differs from
`-D_FORTIFY_SOURCE` in a few ways:

1. **Compile-time warnings only vs. runtime bounds checks**:
   `-Wfortify-source` is purely a compile-time diagnostic flag; it checks
   direct calls to recognized library functions (as well as
   `__builtin___*_chk` builtins) without altering generated code. By contrast,
   `-D_FORTIFY_SOURCE` can also insert runtime bounds checks that abort if an
   overflow is detected at runtime.

2. **Constant evaluation vs. post-optimization checks**:
   `-Wfortify-source` is enabled by default even at `-O0` and without
   `-D_FORTIFY_SOURCE`, making warnings fast, consistent across build
   configurations, and free of false positives from control-flow
   transformations (such as jump threading or loop unrolling). Conversely,
   `-Wfortify-source` only diagnoses calls whose buffer and access sizes can be
   constant-evaluated directly, whereas `-D_FORTIFY_SOURCE` (when compiled with
   `-O1` or higher) can diagnose additional overflows at compile time after
   optimizations (for example, via `warning` or `error` function attributes).

3. **Function coverage**:
   `-Wfortify-source` does not yet cover the full set of functions that
   `-D_FORTIFY_SOURCE` does, though expanding coverage is a
   [work in progress](https://github.com/llvm/llvm-project/issues/142230).
