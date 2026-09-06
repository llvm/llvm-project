# LeakSanitizer

```{contents}
:local: true
```

## Introduction

LeakSanitizer is a run-time memory leak detector. It can be combined with
{doc}`AddressSanitizer` to get both memory error and leak detection, or
used in a stand-alone mode. LSan adds almost no performance overhead
until the very end of the process, at which point there is an extra leak
detection phase.

## Usage

{doc}`AddressSanitizer`: integrates LeakSanitizer and enables it by default on
supported platforms.

```console
$ cat memory-leak.c
#include <stdlib.h>
void *p;
int main() {
  p = malloc(7);
  p = 0; // The memory is leaked here.
  return 0;
}
% clang -fsanitize=address -g memory-leak.c ; ASAN_OPTIONS=detect_leaks=1 ./a.out
==23646==ERROR: LeakSanitizer: detected memory leaks
Direct leak of 7 byte(s) in 1 object(s) allocated from:
    #0 0x4af01b in __interceptor_malloc /projects/compiler-rt/lib/asan/asan_malloc_linux.cc:52:3
    #1 0x4da26a in main memory-leak.c:4:7
    #2 0x7f076fd9cec4 in __libc_start_main libc-start.c:287
SUMMARY: AddressSanitizer: 7 byte(s) leaked in 1 allocation(s).
```

To use LeakSanitizer in stand-alone mode, link your program with
`-fsanitize=leak` flag. Make sure to use `clang` (not `ld`) for the
link step, so that it would link in proper LeakSanitizer run-time library
into the final executable.

### Double-free detection

Stand-alone LeakSanitizer can optionally report a call to `free` on an
allocation that has already been freed. The check is disabled by default;
enable it at run time through `LSAN_OPTIONS`:

```console
$ clang -g -O0 -fno-omit-frame-pointer -fsanitize=leak double-free.c -o double-free
$ LSAN_OPTIONS=detect_double_free=1 ./double-free
==1234==ERROR: LeakSanitizer: attempting double-free on 0x504000000010 in thread T0:
The second free occurred here:
...
The first free occurred here:
...
The memory was allocated here:
...
SUMMARY: LeakSanitizer: double-free
```

A double free is undefined behavior, so build a reproducer at `-O0`: an
optimizing compiler may delete the second `free` before the runtime can
observe it.

Validating the freed pointer is a prerequisite for the check, so while the
option is enabled a `free` of a pointer the allocator never returned is
reported as well, as `bad-free`. Both reports are fatal.
`double_free_max_entries` bounds the state kept for large allocations, whose
metadata does not survive the first free; its default is `65536`.

The option applies to stand-alone LeakSanitizer only, and is not supported on
macOS or NetBSD. Under AddressSanitizer or HWAddressSanitizer the allocator
comes from that tool; use its own double-free diagnostics instead.

## Security Considerations

LeakSanitizer is a bug detection tool and its runtime is not meant to be
linked against production executables. While it may be useful for testing,
LeakSanitizer's runtime was not developed with security-sensitive
constraints in mind and may compromise the security of the resulting executable.

## Supported Platforms

- Android
- Fuchsia
- Linux
- macOS
- NetBSD

## More Information

[https://github.com/google/sanitizers/wiki/AddressSanitizerLeakSanitizer](https://github.com/google/sanitizers/wiki/AddressSanitizerLeakSanitizer)

