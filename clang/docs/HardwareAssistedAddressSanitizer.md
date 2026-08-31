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

- Out-of-bounds accesses (heap, stack, and globals)
- Use-after-free
- Use-after-return and use-after-scope
- Double-free and invalid free

## Supported Platforms

- **Linux AArch64** (kernel 5.4+ with tagged address ABI)
- **Android AArch64**
- **Linux RISC-V 64**
- **Linux x86_64** (experimental)

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
==12345==ERROR: HWAddressSanitizer: tag-mismatch on address 0xef7a2b910005 at pc 0xaaaacb240898
READ of size 1 at 0xef7a2b910005 tags: 0xef/0x7b (ptr/mem)
    #0 0xaaaacb240898 in main example.c:6

Cause: use-after-free
0xef7a2b910005 is located 5 bytes inside a 10-byte region [0xef7a2b910000,0xef7a2b91000a)
freed by thread T0 here:
    #0 0xaaaacb2046df in free ...
    #1 0xaaaacb24088c in main example.c:5

previously allocated by thread T0 here:
    #0 0xaaaacb204460 in malloc ...
    #1 0xaaaacb24087c in main example.c:4
```

## Flags

Runtime flags can be passed using the `HWASAN_OPTIONS` environment variable:

```console
$ HWASAN_OPTIONS="halt_on_error=0" ./a.out
```

Run with `HWASAN_OPTIONS="help=1"` to see all available options.

## Disabling Instrumentation

Functions can be excluded from instrumentation using:

```c
__attribute__((no_sanitize("hwaddress")))
```

Conditional compilation is supported via `__has_feature(hwaddress_sanitizer)`.

## More Information

For details on the design, algorithm, and implementation of HWASan, see
{doc}`HardwareAssistedAddressSanitizerDesign`.
