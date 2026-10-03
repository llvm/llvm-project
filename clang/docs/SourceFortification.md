# Source Fortification

## Introduction

Source fortification (commonly enabled via the
[`_FORTIFY_SOURCE`](https://sourceware.org/glibc/manual/latest/html_node/Source-Fortification.html)
macro in C standard libraries such as glibc and Android's Bionic) hardens calls
to standard C and POSIX library functions against buffer overflows and invalid
arguments using a combination of compile-time diagnostics and runtime bounds
checks.

Clang provides built-in compile-time fortification diagnostics under
{ref}`-Wfortify-source` (and related warning flags), as well as the underlying
builtins and attributes used by fortified C library headers.

## Compile-Time Diagnostics (`-Wfortify-source`)

`-Wfortify-source` is enabled by default in Clang and warns at compile time in
the frontend when calls to supported C library or POSIX functions have provably
out-of-bounds buffer arguments, truncated formatted output, or invalid constant
arguments:

1. **Destination buffer overflows and format truncation**: Diagnoses when a
   write operation will always overflow the destination buffer, when an
   explicit size argument exceeds the known size of the destination buffer, or
   when formatted output will always be truncated:
   - `<string.h>` / `<strings.h>`: `memcpy`, `memmove`, `memset`, `mempcpy`,
     `bcopy`, `bzero`, `strcpy`, `stpcpy`, `strcat`, `strncpy`, `stpncpy`,
     `strncat`, `strlcpy`, `strlcat` (and their `__builtin_` variants).
   - `<stdio.h>`: `sprintf`, `snprintf`, `vsnprintf` (format overflow and
     truncation are also controlled by {ref}`-Wformat-overflow` and
     {ref}`-Wformat-truncation`), `scanf`, `fscanf`, `sscanf`, `fgets`, `fread`.
   - `<poll.h>`: `poll`, `ppoll`, `ppoll64`.
   - `<sys/socket.h>`: `recv`, `recvfrom`.

2. **Source buffer overreads**: Diagnoses when an explicit size argument exceeds
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
  truncation warnings for `sprintf`, `snprintf`, and `vsnprintf`.
- {ref}`-Wbuiltin-memcpy-chk-size`: Diagnoses when the explicit byte count
  passed to most `__builtin___*_chk` functions exceeds the destination object
  size.
- {ref}`-Wstringop-overread`: Diagnoses when memory functions (such as
  `memcpy`, `memmove`, `mempcpy`, `bcopy`, `memchr`, `memcmp`, and `bcmp`) read
  past the end of the source buffer.

## Differences from GCC and glibc `_FORTIFY_SOURCE`

Clang's fortification implementation differs from GCC and glibc in several
important ways:

1. **Frontend semantic checks vs. middle-end analysis**:
   In GCC with glibc, compile-time fortification warnings rely on C library
   headers rewriting calls into `__builtin___*_chk` intrinsics when
   `_FORTIFY_SOURCE` is defined and optimization (`-O1` or higher) is enabled,
   with diagnostics emitted by middle-end optimization passes. In Clang,
   `-Wfortify-source` is a frontend (`Sema`) check enabled by default even at
   `-O0` and without `-D_FORTIFY_SOURCE`, checking direct calls to recognized
   library functions as well as `__builtin___*_chk` builtins.
   - **Benefits**: Warnings are fast, consistent across optimization levels
     (including `-O0` debug builds), independent of C library header
     fortification, and avoid false positives from middle-end control-flow
     transformations (such as jump threading or loop unrolling).
   - **Limitations**: Because checks run in the frontend before inlining and
     value-range propagation, Clang only diagnoses calls where the buffer size
     and access size can be constant-evaluated in the current function (or
     propagated via `pass_object_size`), whereas GCC's middle-end can detect
     overflows exposed after inlining or value-range analysis.

2. **Function coverage**:
   C libraries such as glibc and Bionic fortify a broader set of libc and POSIX
   functions under `_FORTIFY_SOURCE` than Clang's `-Wfortify-source` currently
   diagnoses on unfortified calls (for example, additional functions in
   `<unistd.h>`, `<stdio.h>`, `<sys/socket.h>`, and `<wchar.h>`;
   see [tracking issue #142230](https://github.com/llvm/llvm-project/issues/142230)).

3. **Fortified wrapper functions and attributes**:
   Clang lowers `__builtin_object_size` before middle-end inlining and does not
   support GCC's `__builtin_va_arg_pack` or `__builtin_va_arg_pack_len` in
   inline wrapper functions. To support fortified inline wrappers in C library
   headers (such as Bionic), Clang provides the `pass_object_size` /
   `pass_dynamic_object_size` and `diagnose_as_builtin` attributes (see
   {doc}`AttributeReference` and {doc}`LanguageExtensions`). When a parameter is
   annotated with `pass_object_size(type)`, `-Wfortify-source` evaluates the
   argument's buffer size using the specified `type` (`0`–`3`).
