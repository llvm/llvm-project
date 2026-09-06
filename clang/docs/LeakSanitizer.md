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

## Suppressions

LeakSanitizer reports can be suppressed if you encounter leaks in third-party
libraries or known locations that cannot be fixed immediately.

### Suppression Format

Each suppression rule is specified on its own line in the form:

```text
leak:<pattern>
```

A memory leak is suppressed if `<pattern>` matches any function name, source
file name, or library/module name in the symbolized stack trace of the leak
report. Wildcards (`*`) are supported, and lines starting with `#` are treated
as comments.

Example suppression rules:

```text
# Suppress leak by function name (supports wildcards)
leak:MyKnownLeakyFunction
leak:*LeakyNamespace::*

# Suppress leak by source file name
leak:third_party/leaky_library.cpp

# Suppress leak by shared library / module name
leak:libcrypto.so
```

### Runtime Suppressions

To specify a suppressions file at runtime, pass its path via the `suppressions`
flag in the `LSAN_OPTIONS` environment variable:

```console
$ LSAN_OPTIONS="suppressions=MyLSan.supp" ./a.out
```

(When running LeakSanitizer as part of AddressSanitizer, `LSAN_OPTIONS` is still
used to pass LeakSanitizer-specific flags and suppressions.)

### Compile-time Default Suppressions

You can embed default suppressions directly into your executable at compile/link
time by defining the `__lsan_default_suppressions` function in your source code:

```c++
#include <sanitizer/lsan_interface.h>

extern "C" const char *__lsan_default_suppressions() {
  return "leak:MyKnownLeakyFunction\n"
         "leak:third_party/leaky_library.cpp\n"
         "leak:libcrypto.so\n";
}
```

Both default suppressions and suppressions passed in the file via
`LSAN_OPTIONS="suppressions=..."` will be parsed and applied.

### Programmatic Disabling

LeakSanitizer provides fine-grained programmatic control over leak detection
via `<sanitizer/lsan_interface.h>`:

- **Disable around specific code blocks**: Allocations made between calls to
  `__lsan_disable()` and `__lsan_enable()` will not be reported as leaks. This
  disabling is thread-local and only affects allocations made by the calling thread.
  In C++, you can use the RAII wrapper `__lsan::ScopedDisabler`:

  ```c++
  #include <sanitizer/lsan_interface.h>

  void foo() {
    __lsan::ScopedDisabler disabler;
    // Allocations made here will not be reported as leaks.
    leaky_third_party_init();
  }
  ```

- **Ignore specific objects**: `__lsan_ignore_object(const void *p)` marks the
  heap object pointed to by `p` (and anything reachable from it) as a non-leak.

- **Register custom root regions**: `__lsan_register_root_region(const void *p, size_t size)`
  and `__lsan_unregister_root_region(const void *p, size_t size)` register memory
  areas (such as custom memory pools or mapped regions) to be scanned for live
  pointers during leak checking.

- **Explicit leak checking**: `__lsan_do_leak_check()` triggers an immediate
  leak check. If leaks are detected and the `exitcode` flag is non-zero (default),
  the process terminates; otherwise, it returns normally. Calling this function
  disables subsequent automatic leak checks at process exit.
  `__lsan_do_recoverable_leak_check()` performs a leak check and returns `0` if
  no leaks were detected (or if leak detection is disabled), and `1` if leaks
  were found. It prints a report without terminating the process or disabling
  the end-of-process check.

- **Disable leak checking entirely**: Define `__lsan_is_turned_off()` to return
  `1` to disable leak checking for the program.

## Flags and Options

Runtime flags can be passed to LeakSanitizer via the `LSAN_OPTIONS` environment
variable:

```console
$ LSAN_OPTIONS="print_suppressions=0:report_objects=1" ./a.out
```

To see the full list of available flags, run an instrumented binary with
`LSAN_OPTIONS="help=1"`.

Flags passed via the `LSAN_OPTIONS` environment variable take precedence over
compile-time default options.

### Compile-time Default Options

Default options can also be specified at compile/link time by defining
`__lsan_default_options`:

```c++
#include <sanitizer/lsan_interface.h>

extern "C" const char *__lsan_default_options() {
  return "print_suppressions=0:report_objects=1";
}
```

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
