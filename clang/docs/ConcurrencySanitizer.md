# ConcurrencySanitizer

## Introduction

ConcurrencySanitizer (CSan) is a data race detector for CPU and GPU programs.
It consists of an LLVM instrumentation pass and run-time libraries in
compiler-rt.

Unlike ThreadSanitizer, CSan does not maintain a happens-before model or shadow
state for every memory location. Memory accesses are sampled at random and
stalled to detect unordered accesses. Importantly, this method has **no false
positives** and **fixed memory overhead.** However, the probabalistic nature
means that many runs are required to state confidently that the application is
not racy.

## How to build

Build LLVM/Clang with [CMake](https://llvm.org/docs/CMake.html). To enable
offloading support, the easiest configuration is provided in a CMake cache file.

```sh
cmake ../llvm -G Ninja                       \
    -C ../offload/cmake/caches/Offload.cmake \
    -DCMAKE_BUILD_TYPE=Release               \
    -DCMAKE_INSTALL_PREFIX=<PATH>
```

## Supported platforms

CSan currently supports:

- Linux on x86-64.
- AMDGPU devices on Linux through the HSA runtime.

Support for other hosts and GPU targets is not currently implemented.

## Usage

Compile and link the complete program with `-fsanitize=concurrency`. Use `-g`
to include source locations in reports and `-O1` or higher for representative
optimized code.

For a CPU program:

```console
$ clang++ -fsanitize=concurrency -g -O1 race.cpp -pthread
$ ./a.out
```

For a HIP program:

```console
$ clang++ -x hip --offload-arch=gfx1030 -fsanitize=concurrency \
    -g -O1 race.hip
$ ./a.out
```

For an OpenMP offload program:

```console
$ clang++ -fopenmp --offload-arch=gfx1030 -fsanitize=concurrency \
    -g -O1 race.cpp
$ ./a.out
```

When CSan detects a race, it writes a report to standard error. A watchpoint
report normally identifies both conflicting accesses:

```text
WARNING: ConcurrencySanitizer: data race
  Write of size 4 at 0x...:
    #0 update race.cpp:12
  Previous write of size 4 at 0x...:
    #0 update race.cpp:13
```

A value change without a matching instrumented access is reported as a
`data race of unknown origin` and contains only the sampled access. GPU reports
also identify the block, thread, and lane and may name the global variable
containing the raced address.

## How it works

The instrumentation marks memory accesses with a runtime function. The runtime
then samples these accesses. Sampling is fundamentally stalling a given value
while checking if any other threads have touched it during that window. This is
done using a **watchpoint** table and by **value comparison**. Each probed
access sets up a tripwire and checks later if another thread triggered it like
in the following pseudocode:

```c
static u64 watchpoints[N]; // Hash-indexed, zero is empty.

// Emitted before the access to simulate a stalled operation.
void check_access(volatile void *addr, u32 size, u32 type) {
    // Every access probes. A read conflicts only with a watched write, a
    // write conflicts with either.
    if (u64 *wp = find_watchpoint(addr, size, type))
        consume(wp, this_pc()); // Hand our location to the owner.

    if (!should_sample()) // 1-in-N chance for a non-atomic access.
        return;

    u64 *wp = arm_watchpoint(addr, size, type);
    if (!wp) // Slot is already taken.
        return;

    auto old = read(addr, size);
    delay(ctx.rand()); // Randomized, bounded.
    auto new = read(addr, size);

    if (void *peer = disarm(wp))
        report_race(addr, this_pc(), peer);
    else if (new != old)
        report_race(addr, this_pc(), UNKNOWN);
}
```

The probability of detecting a race and the overhead of the sanitizer is
directly related to the probability of sampling and the length of the delay. The
watchpoint table can be omitted to make a minimal detector with less coverage.

## `__SANITIZE_CONCURRENCY__`

Code can test whether CSan instrumentation is enabled:

```c
#if defined(__SANITIZE_CONCURRENCY__)
// Code built with ConcurrencySanitizer.
#endif
```

## Disabling instrumentation

Use `no_sanitize("concurrency")` to disable CSan memory-access instrumentation
for a function:

```c
__attribute__((no_sanitize("concurrency")))
void uninstrumented_function() {
  // ...
}
```

The `disable_sanitizer_instrumentation` attribute disables all sanitizer
instrumentation and takes precedence over `no_sanitize` attributes. Use either
form carefully: uninstrumented accesses cannot consume watchpoints and may
reduce detection or produce reports of unknown origin.

## Run-time flags

CSan reads options from the `CSAN_OPTIONS` environment variable:

```console
$ CSAN_OPTIONS="skip_watch=1000:udelay=200" ./a.out
```

The CSan-specific host options are:

- `skip_watch` (default: `4000`): Approximate number of accesses skipped by a
  thread before arming another watchpoint.
- `udelay` (default: `80`): Microseconds to delay after arming a watchpoint.
- `halt_on_error` (default: `false`): Terminate after the first host report.

These options currently control the host detector. The AMDGPU sampling rate and
delay interval are fixed. Common sanitizer options still control facilities
such as symbolization and report formatting on the host side of GPU reporting.

Use `CSAN_OPTIONS=help=1` to print all CSan and sanitizer-common options.
Programs may provide host defaults by defining:

```c++
extern "C" const char *__csan_default_options() {
  return "skip_watch=1000:udelay=200";
}
```

Environment options take precedence over `__csan_default_options`.

## Limitations

- CSan is probabilistic. It detects races that overlap a sampled observation
  window rather than proving that a program is race-free.
- CSan does not implement ThreadSanitizer's happens-before model, lock-order
  analysis, thread history, or synchronization diagnostics.
- CSan cannot be combined with several other full sanitizers, including
  ThreadSanitizer and AddressSanitizer.

## Security considerations

ConcurrencySanitizer is a testing tool. Its runtime is not intended for
production executables and was not developed under security-sensitive runtime
constraints. Instrumented programs deliberately change scheduling and reserve
additional memory.

## Current status

ConcurrencySanitizer is experimental. Its interface, reports, supported targets,
sampling policy, and run-time options may change. The compiler-rt test suite can
be run with:

```console
$ ninja check-csan
```

Minimized test cases and reports from real CPU and GPU applications are
welcome.
