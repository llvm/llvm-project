# ThreadSanitizer

## Introduction

ThreadSanitizer is a tool that detects data races. It consists of a compiler
instrumentation module and a run-time library. Typical slowdown introduced by
ThreadSanitizer is about **5x-15x**. Typical memory overhead introduced by
ThreadSanitizer is about **5x-10x**.

## How to build

Build LLVM/Clang with [CMake](https://llvm.org/docs/CMake.html).

## Supported Platforms

ThreadSanitizer is supported on the following OS:

- Android aarch64, x86_64
- Darwin arm64, x86_64
- FreeBSD
- Linux aarch64, x86_64, powerpc64, powerpc64le
- NetBSD

Support for other 64-bit architectures is possible, contributions are welcome.
Support for 32-bit platforms is problematic and is not planned.

## Usage

Simply compile and link your program with `-fsanitize=thread`. To get a
reasonable performance add `-O1` or higher. Use `-g` to get file names
and line numbers in the warning messages.

Example:

```console
% cat projects/compiler-rt/lib/tsan/lit_tests/tiny_race.c
#include <pthread.h>
int Global;
void *Thread1(void *x) {
  Global = 42;
  return x;
}
int main() {
  pthread_t t;
  pthread_create(&t, NULL, Thread1, NULL);
  Global = 43;
  pthread_join(t, NULL);
  return Global;
}

$ clang -fsanitize=thread -g -O1 tiny_race.c
```

If a bug is detected, the program will print an error message to stderr.
Currently, ThreadSanitizer symbolizes its output using an external
`addr2line` process (this will be fixed in future).

```bash
% ./a.out
WARNING: ThreadSanitizer: data race (pid=19219)
  Write of size 4 at 0x7fcf47b21bc0 by thread T1:
    #0 Thread1 tiny_race.c:4 (exe+0x00000000a360)

  Previous write of size 4 at 0x7fcf47b21bc0 by main thread:
    #0 main tiny_race.c:10 (exe+0x00000000a3b4)

  Thread T1 (running) created at:
    #0 pthread_create tsan_interceptors.cc:705 (exe+0x00000000c790)
    #1 main tiny_race.c:9 (exe+0x00000000a3a4)
```

## `__has_feature(thread_sanitizer)`

In some cases one may need to execute different code depending on whether
ThreadSanitizer is enabled.
{ref}`__has_feature <langext-__has_feature-__has_extension>` can be used for
this purpose.

```c
#if defined(__has_feature)
#  if __has_feature(thread_sanitizer)
// code that builds only under ThreadSanitizer
#  endif
#endif
```

## `__attribute__((no_sanitize("thread")))`

Some code should not be instrumented by ThreadSanitizer. One may use the
function attribute `no_sanitize("thread")` to disable instrumentation of plain
(non-atomic) loads/stores in a particular function. ThreadSanitizer still
instruments such functions to avoid false positives and provide meaningful stack
traces. This attribute may not be supported by other compilers, so we suggest
to use it together with `__has_feature(thread_sanitizer)`.

## `__attribute__((disable_sanitizer_instrumentation))`

The `disable_sanitizer_instrumentation` attribute can be applied to functions
to prevent all kinds of instrumentation. As a result, it may introduce false
positives and incorrect stack traces. Therefore, it should be used with care,
and only if absolutely required; for example for certain code that cannot
tolerate any instrumentation and resulting side-effects. This attribute
overrides `no_sanitize("thread")`.

## Interaction of Inlining with Disabling Sanitizer Instrumentation

- A `no_sanitize` function will not be inlined heuristically by the compiler into a sanitized function.
- An `always_inline` function will adopt the instrumentation status of the function it is inlined into.
- Forcibly combining `no_sanitize` and `__attribute__((always_inline))` is not supported, and will often lead to unexpected results. To avoid mixing these attributes, use:

```c
// Note, __has_feature test for sanitizers is deprecated, and Clang will support __SANITIZE_<sanitizer>__ similar to GCC.
#if __has_feature(thread_sanitizer) || defined(__SANITIZE_THREAD__) || ... <other sanitizers>
#define ALWAYS_INLINE_IF_UNINSTRUMENTED
#else
#define ALWAYS_INLINE_IF_UNINSTRUMENTED __attribute__((always_inline))
#endif
```

## Explicit Sanitizer Checks with `__builtin_allow_sanitize_check`

The `__builtin_allow_sanitize_check("thread")` builtin can be used to
conditionally execute code depending on whether ThreadSanitizer checks are
enabled and permitted by the current policy (after inlining). This is
particularly useful for inserting explicit, sanitizer-specific checks around
operations like syscalls or inline assembly, which might otherwise be unchecked
by the sanitizer.

Example:

```c
void __tsan_read8(void *);

inline __attribute__((always_inline))
void my_helper(void *addr) {
  if (__builtin_allow_sanitize_check("thread"))
    __tsan_read8(addr);
  // ... actual logic, e.g. inline assembly ...
  asm volatile ("..." : : "r" (addr) : "memory");
}

void instrumented_function() {
  ...
  my_helper(&shared_data); // checks are active
  ...
}

__attribute__((no_sanitize("thread")))
void uninstrumented_function() {
  ...
  my_helper(&shared_data); // checks are skipped
  ...
}
```

## Ignorelist

ThreadSanitizer supports `src` and `fun` entity types in
{doc}`SanitizerSpecialCaseList`, that can be used to suppress data race reports
in the specified source files or functions. Unlike functions marked with
`no_sanitize("thread")` attribute, ignored functions are not instrumented
at all. This can lead to false positives due to missed synchronization via
atomic operations and missed stack frames in reports.

Example:

```bash
# Ignore exactly this function (the names are mangled in C++)
fun:_Z8MyFooBarv
# Ignore all functions containing MyFooBar
fun:*MyFooBar*
# Ignore the whole file
src:file_with_tricky_code.cc
```

## Limitations

- ThreadSanitizer uses more real memory than a native run. At the default
  settings the memory overhead is 5x plus 1Mb per each thread. Settings with 3x
  (less accurate analysis) and 9x (more accurate analysis) overhead are also
  available.

- ThreadSanitizer maps (but does not reserve) a lot of virtual address space.
  This means that tools like `ulimit` may not work as usually expected.

- Libc/libstdc++ static linking is not supported.

- Non-position-independent executables are not supported. Therefore, the
  `fsanitize=thread` flag will cause Clang to act as though the `-fPIE`
  flag had been supplied if compiling without `-fPIC`, and as though the
  `-pie` flag had been supplied if linking an executable.

- ThreadSanitizer generally requires all code to be compiled with
  `-fsanitize=thread`. If some code (such as pre-compiled dynamic libraries)
  is not compiled with the flag, TSan may fail to detect races or may report
  false positives. Refer to the `ignore_interceptors_accesses` and
  `ignore_noninstrumented_modules` run-time options to work around issues
  arising from non-instrumented modules.

- On Linux, disabling ASLR may cause ThreadSanitizer to fail to allocate shadow
  memory, printing the error `FATAL: ThreadSanitizer can not mmap the shadow memory (something is mapped at ...)`.
  Note that GDB disables ASLR by default. To debug ThreadSanitizer binaries under
  GDB, configure it to preserve ASLR by running:

  ```console
  $ gdb -ex 'set disable-randomization off' --args ./a.out
  ```

## Security Considerations

ThreadSanitizer is a bug detection tool and its runtime is not meant to be
linked against production executables. While it may be useful for testing,
ThreadSanitizer's runtime was not developed with security-sensitive
constraints in mind and may compromise the security of the resulting executable.

## Current Status

ThreadSanitizer is in beta stage. It is known to work on large C++ programs
using pthreads, but we do not promise anything (yet). C++11 threading is
supported with llvm libc++. The test suite is integrated into CMake build
and can be run with `make check-tsan` command.

We are actively working on enhancing the tool --- stay tuned. Any help,
especially in the form of minimized standalone tests is more than welcome.

## Run-time Flags

ThreadSanitizer supports a number of run-time flags, which can be specified
in the `TSAN_OPTIONS` environment variable. Separate flags are separated
with spaces.

Example:

```console
$ TSAN_OPTIONS="history_size=7 halt_on_error=1" ./myprogram
```

The most common run-time flags are:

- `detect_deadlocks` (default: `true`): Controls whether deadlock detection is
  enabled.
- `exitcode` (default: `66`): Override the exit status if an error is found. Note
  that for ThreadSanitizer the default exitcode is 66, unlike other sanitizers.
- `force_seq_cst_atomics` (default: `false`): If set, all atomic operations are
  treated as sequentially consistent (seq_cst). Useful for debugging relaxed atomic
  ordering bugs.
- `halt_on_error` (default: `false`): Exit after the first reported error.
- `history_size` (default: `0`): Controls per-thread history size, which
  determines how many previous memory accesses are remembered per thread. If
  you see "failed to restore the stack" reports, try increasing this flag
  (values from 0 to 7 are supported).
- `ignore_interceptors_accesses` (default: `false` on Linux/Windows, `true`
  on Apple platforms): Ignore reads and writes from all interceptors.
- `ignore_noninstrumented_modules` (default: `false` on Linux/Windows, `true`
  on Apple platforms): Ignore interceptors called from non-instrumented modules.
- `log_path` (default: `""`): If set, write logs to `log_path.pid` instead of
  stderr. Special values are `stdout` and `stderr`.
- `print_full_thread_history` (default: `false`): Print creation stack traces for
  all threads involved in the report and their ancestor threads back to the main thread.
- `report_atomic_races` (default: `true`): Report races between atomic and
  plain memory accesses.
- `suppressions` (default: `""`): Path to a suppressions file.

To see the complete list of flags and their descriptions, run an instrumented
binary with the `help=1` option:

```console
$ TSAN_OPTIONS="help=1" ./myprogram
```

You can also refer to the source declarations in the LLVM repository under
`compiler-rt/lib/tsan/rtl/tsan_flags.inc` (ThreadSanitizer-specific flags)
and `compiler-rt/lib/sanitizer_common/sanitizer_flags.inc` (common sanitizer
flags).

## Suppressions

If you have a data race or thread leak that you are already aware of but cannot
fix right away (e.g., in a third-party library), you can suppress the reports
at run-time using a suppressions file.

Specify the suppressions file path via the `suppressions` flag in the
`TSAN_OPTIONS` environment variable:

```console
$ TSAN_OPTIONS="suppressions=/path/to/suppressions.supp" ./myprogram
```

Each non-empty line of the suppressions file represents one suppression of the
form:

```
suppression_type:suppression_pattern
```

The supported `suppression_type` values are:

- `race`: Suppresses data race reports. The pattern is matched against function
  names, source file names, or global variable names in the stacks of the report.
- `thread`: Suppresses thread leak reports. The pattern is matched against the
  name of the leaked thread.
- `called_from_lib`: Suppresses reports if the call originated from a specific
  non-instrumented library.

The pattern can contain the `*` wildcard, which matches any substring. By
default, `*` is automatically prepended and appended to each pattern. You can
use `^` and `$` to anchor patterns to the beginning and end of the string.
Lines starting with `#` are treated as comments.

Example of a suppressions file:

```text
# Suppress data races in a third-party library 'foobar'
race:foobar
# Suppress data races in a specific function
race:NuclearRocket::Launch
# Suppress data races in a specific source file
race:src/surgery/laser_scalpel.cc
# Suppress data races on a specific global variable
race:global_var
# Suppress a leaked thread by name
thread:MonitoringThread
# Suppress warnings called from an uninstrumented library
called_from_lib:libzmq.so
```

## Adaptive Delay

### Overview

Adaptive Delay is an optional ThreadSanitizer feature that injects delays at
synchronization points to explore novel thread interleavings and increase the
likelihood of exposing data races. By perturbing thread scheduling, adaptive
delay creates more opportunities for concurrent accesses to shared data,
improving race detection.

Adaptive delay is particularly useful for:

- Detecting races in rarely-executed thread interleavings or code paths
- Testing parallel data structures and algorithms

When enabled, adaptive delay maintains a configurable time budget to balance
race exposure against performance overhead. The delays can be

> - random amount of spin cycles
> - a single yield to the OS scheduler
> - random usleep

The strategy prioritizes high-value synchronization points:

- Relaxed atomic operations receive cheap delays (spin cycles) with low sampling
- Synchronizing atomic operations (acquire/release/seq_cst) receive moderate
  delays with higher sampling
- Mutex and thread lifecycle operations receive the longest delays with highest
  sampling

The delays focus on synchronization points with clear happens-before relationships,
as those are most likely to expose data races.

### Enabling Adaptive Delay

Adaptive delay is disabled by default. Enable it by setting the
`enable_adaptive_delay` flag:

```console
$ TSAN_OPTIONS=enable_adaptive_delay=1 ./myapp
```

### Configuration Options

```{list-table} Adaptive Delay Options
:name: adaptive-delay-options-table
:header-rows: 1
:widths: 35 10 15 40

* - Flag
  - Type
  - Default
  - Description
* - `enable_adaptive_delay`
  - bool
  - false
  - Enable adaptive delay injection to expose data races.
* - `adaptive_delay_aggressiveness`
  - int
  - 25
  - Controls delay injection intensity for race detection. Higher values inject
    more delays to expose races. Value must be greater than 0. Suggested values:
    10 (minimal), 50 (moderate), 200 (aggressive). This is a tuning parameter;
    actual overhead varies by workload and platform.
* - `adaptive_delay_relaxed_sample_rate`
  - int
  - 10000
  - Sample 1 in N relaxed atomic operations for delay injection. Relaxed atomics
    have minimal synchronization, so sampling helps avoid excessive overhead.
* - `adaptive_delay_sync_atomic_sample_rate`
  - int
  - 100
  - Sample 1 in N acquire/release/seq_cst atomic operations for delay injection.
    These synchronizing atomics are more likely to expose races, so are sampled
    more often.
* - `adaptive_delay_mutex_sample_rate`
  - int
  - 10
  - Sample 1 in N mutex/condition variable operations for delay injection. Mutex
    ops are high-value synchronization points and are sampled frequently.
* - `adaptive_delay_max_atomic`
  - string
  - `"sleep_us=50"`
  - Maximum delay for atomic operations. Format: `"spin=N"` (N spin cycles,
    1 <= N <= 10,000), `"yield"` (one yield to the OS), or `"sleep_us=N"`
    (up to N microseconds). The delay is randomly chosen up to the specified
    maximum N.
* - `adaptive_delay_max_sync`
  - string
  - `"sleep_us=500"`
  - Maximum delay for synchronization operations (mutex and thread lifecycle
    operations). Format: same as `adaptive_delay_max_atomic`. Typically set
    longer than atomic delays since these operations involve waking blocked threads
    and may be more likely to expose races.
```

### Examples

Enable adaptive delay with moderate aggressiveness:

```console
$ TSAN_OPTIONS=enable_adaptive_delay=1:adaptive_delay_aggressiveness=50 ./myapp
```

Enable aggressive delay injection:

```console
$ TSAN_OPTIONS=enable_adaptive_delay=1:adaptive_delay_aggressiveness=200 ./myapp
```

Increase sampling frequency for mutex operations:

```console
$ TSAN_OPTIONS=enable_adaptive_delay=1:adaptive_delay_mutex_sample_rate=5 ./myapp
```
