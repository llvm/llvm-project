# Hardware-assisted AddressSanitizer

```{contents}
:local: true
```

## Introduction

Hardware-assisted AddressSanitizer (HWASan) is a memory error detector similar
to {doc}`AddressSanitizer`, but based on memory tagging and address tagging
(such as Top-Byte Ignore on AArch64). It provides similar bug detection with
significantly lower memory overhead (~10–20%).

HWASan can detect:

- Out-of-bounds accesses (heap, stack; globals tagging is not supported on all platforms)
- Use-after-free
- Use-after-return and use-after-scope
- Double-free and invalid free

## Supported Platforms

- **Linux AArch64** (kernel 5.4+ with tagged address ABI; globals tagging is not implemented)
- **Android AArch64**
- **Linux RISC-V 64**
- **Linux x86_64** (experimental; requires CPU with Intel LAM and kernel support)

## Usage

Compile and link your program with `-fsanitize=hwaddress`:

```console
% cat example.c
#include <stdlib.h>

int main() {
  char *volatile x = (char *)malloc(10);
  free(x);
  return x[5]; // Use-after-free!
}

% clang -O1 -g -fsanitize=hwaddress example.c
% ./a.out
==3304567==ERROR: HWAddressSanitizer: tag-mismatch on address 0xe469fffe0005 at pc 0xc05830a6e228
READ of size 1 at 0xe469fffe0005 tags: 60/7a (ptr/mem) in thread T0
    #0 0xc05830a6e228 in main example.c:6:10
    #1 0xf85b82ab2f18  (/usr/lib/aarch64-linux-gnu/libc.so.6+0x22f18)
    #2 0xf85b82ab3058 in __libc_start_main (/usr/lib/aarch64-linux-gnu/libc.so.6+0x23058)
    #3 0xc05830a231ec in _start (a.out+0x431ec)

[0xe469fffe0000,0xe469fffe0010) is a small unallocated heap chunk; size: 16 offset: 5

Cause: use-after-free
0xe469fffe0005 is located 5 bytes inside a 10-byte region [0xe469fffe0000,0xe469fffe000a)
freed by thread T0 here:
    #0 0xc05830a2b5c0 in free (a.out+0x4b5c0)
    #1 0xc05830a6e1f8 in main example.c:5:3
    #2 0xf85b82ab2f18  (/usr/lib/aarch64-linux-gnu/libc.so.6+0x22f18)
    #3 0xf85b82ab3058 in __libc_start_main (/usr/lib/aarch64-linux-gnu/libc.so.6+0x23058)
    #4 0xc05830a231ec in _start (a.out+0x431ec)

previously allocated by thread T0 here:
    #0 0xc05830a2bc38 in malloc (a.out+0x4bc38)
    #1 0xc05830a6e1ec in main example.c:4:30
    #2 0xf85b82ab2f18  (/usr/lib/aarch64-linux-gnu/libc.so.6+0x22f18)
    #3 0xf85b82ab3058 in __libc_start_main (/usr/lib/aarch64-linux-gnu/libc.so.6+0x23058)
    #4 0xc05830a231ec in _start (a.out+0x431ec)

SUMMARY: HWAddressSanitizer: tag-mismatch example.c:6:10 in main
```

## Flags and Options

Runtime flags can be passed using the `HWASAN_OPTIONS` environment variable:

```console
$ HWASAN_OPTIONS="halt_on_error=0" ./a.out
```

To see the full list of available flags, run an instrumented binary with
`HWASAN_OPTIONS="help=1"`.

Flags passed via the `HWASAN_OPTIONS` environment variable take precedence over
compile-time default options.

### Compile-time Default Options

Default options can be specified at compile/link time by defining
the `__hwasan_default_options` function in your source code:

```c++
#include <sanitizer/hwasan_interface.h>

extern "C" const char *__hwasan_default_options() {
  return "halt_on_error=0";
}
```

### Options Evaluation with Integrated Sanitizers

HWASan works together with {doc}`LeakSanitizer` and {doc}`UndefinedBehaviorSanitizer`:

- `__hwasan_default_options()`, `__lsan_default_options()`, and `__ubsan_default_options()`
  are all evaluated independently by their respective runtimes.
- Similarly, `HWASAN_OPTIONS`, `LSAN_OPTIONS`, and `UBSAN_OPTIONS` environment
  variables are parsed separately by their respective runtimes.

## Disabling Instrumentation

Functions can be excluded from instrumentation using:

```c
__attribute__((no_sanitize("hwaddress")))
```

Conditional compilation is supported via `__has_feature(hwaddress_sanitizer)`.

## More Information

For details on the design, algorithm, and implementation of HWASan, see
{doc}`HardwareAssistedAddressSanitizerDesign`.
