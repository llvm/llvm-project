(stdfix)=

# Fixed-Point Arithmetic

## Standards and Goals

- `stdfix.h` is specified in [ISO/IEC TR 18037:2008](https://www.iso.org/standard/51126.html),
  C extensions to support embedded processors
  ([publicly available specification](https://standards.iso.org/ittf/PubliclyAvailableStandards/c051126_ISO_IEC_TR_18037_2008.zip)).
- Our goal is to implement a complete set of math functions for fixed-point
  types, most of which are not included in ISO/IEC TR 18037:2008.  Our math
  functions for fixed-point types are modelled after the C99/C23 math functions
  for floating-point types.

## Source Location

- The main source for fixed-point functions is located at
  [libc/src/stdfix](https://github.com/llvm/llvm-project/tree/main/libc/src/stdfix),
  with internal support utilities under
  [libc/src/\_\_support/fixed_point](https://github.com/llvm/llvm-project/tree/main/libc/src/__support/fixed_point).

## Implementation Status

- Per-precision function, macro, and type status tables: {doc}`../headers/stdfix`

### Requirements

- Building LLVM-libc with fixed-point arithmetic support requires a compiler
  that supports the native fixed-point types `_Fract` and `_Accum` in C++.
- Consuming the generated `stdfix.h` header requires compiler support for
  `_Fract` and `_Accum` in C or C++.
- Compiler support is checked at the top of
  [libc/include/llvm-libc-macros/stdfix-macros.h](https://github.com/llvm/llvm-project/tree/main/libc/include/llvm-libc-macros/stdfix-macros.h).

### Predefined Macros

- `LIBC_COMPILER_HAS_FIXED_POINT` indicates whether the compiler supports
  fixed-point types.
- Predefined precision and range macros from ISO/IEC TR 18037:2008 section
  7.18a.3 are defined in
  [libc/include/llvm-libc-macros/stdfix-macros.h](https://github.com/llvm/llvm-project/tree/main/libc/include/llvm-libc-macros/stdfix-macros.h)
  using the default `typical desktop processor` configuration from Annex A.3
  when compiler predefined macros are absent.

### Formatted I/O

- ISO/IEC TR 18037:2008 section 7.18a.6.7 also specifies fixed-point conversion
  specifiers (`r`, `R`, `k`, `K`) for the `<stdio.h>` `fprintf` and `fscanf`
  families (`LIBC_CONF_PRINTF_DISABLE_FIXED_POINT` controls `printf` support).

### Warnings

This support is currently a work in progress; its headers, macros, and ABI are
still unstable and may change.
