(implementation_guidelines)=

# Implementation Guidelines

{ref}`code_style` covers naming, macros, and formatting. This page covers the
judgment calls that come up while writing an implementation: what to optimize
for, how to use memory, when to check inputs, and how to handle headers.
{ref}`patterns_to_avoid` lists the mistakes and outdated patterns that
reviewers see most often.

## Priorities

When goals conflict, prefer them in this order:

1. **Correctness.** Follow the standard. See {ref}`undefined_behavior` for
   what to do where the standard is silent.
2. **Readability.** Code that is long but obvious is better than code that is
   short and clever. The character classification tables in
   `src/__support/ctype_utils.h` are long, and anyone can see what they do.
3. **Efficiency.** This encompasses latency, throughput, binary size, and memory
   usage. There's no consistent formula for tradeoffs between these, but in
   general baremetal users prioritize binary size and servers prioritize latency
   and throughput.

The memory functions and the math library are exceptions, where performance is
an explicit goal alongside correctness.

Microbenchmarks are useful for finding a direction but are not a goal in and of
themselves. The ideal performance measurement is from sampling real systems,
though that's impractical for most changes. The project does not have a settled
benchmarking setup today. LLVM's
[benchmarking tips](https://llvm.org/docs/Benchmarking.html) describe how to
take more meaningful measurements.

## Memory

- **Allocate on the heap only when necessary.** Many targets have a small
  heap or none. When allocation is needed, follow the allocation rules in
  {ref}`code_style` and handle failure.
- **Stack buffers may be large when the function needs them.** Prefer a
  correct implementation with a generous buffer over a complicated one. Sizes
  can be tuned later.
- **Do not use unbounded recursion.** Baremetal users must be able to
  determine the maximum stack depth statically. Recursion with an explicit
  depth limit is acceptable.

### Allocators

LLVM-libc does not aim to provide its own general-purpose allocator for hosted
systems. On Linux the supported allocator is
[Scudo](https://llvm.org/docs/ScudoHardenedAllocator.html), and users with
specific performance requirements are expected to bring the allocator they
already prefer.

Scudo is not used on baremetal targets. It is designed for hosted systems and
gets its memory from the operating system by mapping pages. A baremetal target
has no operating system to ask, and usually has a small, fixed region of
memory. On baremetal targets LLVM-libc instead provides an optimized allocator
in `src/__support/freelist_heap.h`, which manages the memory between two
symbols that the linker script provides.

## Assembly

LLVM-libc is written in C++. Assembly is used only in the few places where
the stack or individual registers must be controlled exactly, or where an
instruction has no compiler builtin. Assembly written for performance is not
accepted, because it gives up what the compiler provides for C++ code:
diagnostics, sanitizers, coverage and profile instrumentation, and debug
information. {ref}`assembly_policy` has the full policy, including how
patches that contain assembly are reviewed.

## Runtime checks

- **Null pointers.** At the start of an entrypoint, pass each pointer argument
  that would be dereferenced, and for which a null pointer is not a valid
  input, to `LIBC_CRASH_ON_NULLPTR` (`src/__support/macros/null_check.h`). The
  check is lightweight, is controlled by the `LIBC_ADD_NULL_CHECKS` option, and
  exists to make a crash easy to diagnose. Put it in the entrypoint only, not
  in internal code. Do not add the check for a pointer that the libc only
  passes on to the kernel without dereferencing it. The kernel checks such
  pointers safely and returns `EFAULT` for any invalid address, not only for
  a null pointer.
- **Internal invariants.** Use `LIBC_ASSERT`. It compiles away in release
  builds.
- **Hardening.** `LIBC_COPT_HARDENING_MODE` was added for the allocator's
  integrity checks. Extending it to other code needs discussion first.
- **Everything else.** Do not validate inputs beyond what the standard
  requires. {ref}`undefined_behavior` explains the reasoning.

## Headers

- **Include what you use.** Every name a file uses must be made available by
  a header that the file includes directly, not by a transitive inclusion. A
  file that names `mode_t` includes `hdr/types/mode_t.h` itself, even if
  another header it includes happens to provide the type. This applies to
  header files as well as `.cpp` files. The exception is a header whose
  purpose is to include another one, such as a proxy header or a header that
  selects the implementation for the target: include that header, not the
  file it forwards to.
- **Use proxy headers** from `libc/hdr/` for every type and macro that the
  libc also provides publicly. See the header inclusion policy in
  {ref}`code_style`. Compiler-provided headers are tolerated but should also
  go through a proxy header where one exists. The existing `#include
  <pthread.h>` style includes under `libc/src/` are incorrect and should not
  be copied.
- **`src/__support/CPP/` headers** are included through the top-level header,
  for example `src/__support/CPP/type_traits.h` or
  `src/__support/CPP/utility.h`, not through the individual files in the
  subdirectories of the same name. The top-level header is a forwarding
  header of the kind the first point describes.

(feature_test_macros)=

## Feature-test macros in public headers

Public headers do not use feature-test macros such as `_GNU_SOURCE` or
`_POSIX_C_SOURCE` to hide declarations. Do not add `#ifdef` guards of that kind
to a header. The exception is a macro that a standard itself defines for this
purpose, such as `__STDC_WANT_LIB_EXT1__` for Annex K.

Headers are generated from the YAML files in `libc/include/` (see
{ref}`header_generation`), and a generated header declares only the functions
that are enabled for the target. A user who wants a different API surface
generates headers to match it, by pointing the `LIBC_CONFIG_PATH` CMake option
at their own `entrypoints.txt`, `headers.txt`, and configuration (see
{ref}`custom_config_directory`), instead of every user receiving one set of
headers full of conditionals.

Each function in the YAML files lists the `standards` it comes from (`stdc`,
`posix`, `gnu`, `bsd`, `linux`, `llvm_libc_ext`, and so on). Keep that field
accurate. It records where each declaration comes from.

{ref}`writing_public_headers` has the other rules for public headers.

## File headers and comments

Every file touched by a change uses the current LLVM file header: a bare
separator line, the license block, and a `\file` description.

```c++
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of fflush.
///
//===----------------------------------------------------------------------===//
```

Comment only what the code does not already say. A straightforward function
or test needs no introductory paragraph, and a function that is moved keeps
exactly the comments it had. Also, don't number the steps of a function in
its comments (`// 1. ...`, `// 2. ...`). It makes later refactors noisier than
necessary for no tangible benefit.

## Character data

Do not compare against character literals (except for equality), or do
arithmetic on them to classify or convert characters. Use the helpers in
`src/__support/ctype_utils.h` and `src/__support/wctype_utils.h`, which work
for non-ASCII encodings and for both `char` and `wchar_t`. See
{ref}`design_philosophy` for the model behind this.

(math_function_guidelines)=

## Math functions

The {ref}`math page <math>` describes the goals of the math library, and
`libc/src/math/docs/add_math_function.md` is the step-by-step guide for adding
a function. The points below are the ones that most often need stating.

### Accuracy and error reporting

- The default implementation of every math function is expected to be
  correctly rounded in all rounding modes, and to support `errno`, floating
  point exceptions, and signaling NaNs. A signaling NaN input produces a quiet
  NaN and raises `FE_INVALID`.
- The exceptions to that expectation are:
  - a build that selects a different trade-off with the
    `LIBC_CONF_MATH_OPTIMIZATIONS` option, for example
    `LIBC_MATH_SKIP_ACCURATE_PASS`, `LIBC_MATH_NO_ERRNO`, or
    `LIBC_MATH_NO_EXCEPT`. The baremetal and GPU configurations set several
    of these by default;
  - an implementation variant other than the default one (see
    {ref}`math_function_structure`);
  - the GPU-specific math implementations, which have different accuracy
    bounds and are maintained by the GPU maintainers.
- The {doc}`math.h status tables <../headers/math>` record the accuracy of
  each function, including any that are not correctly rounded yet.
- `errno` and floating point exceptions are reported only through
  `fputil::set_errno_if_required` and `fputil::raise_except_if_required`.

(math_function_structure)=

### Structure

- The implementation lives in a header under `src/__support/math/`, and the
  entrypoint is a thin wrapper.
- A function can have several implementations, each in its own namespace
  (`double_eval`, `float_eval`, `float_eval_cr_rn`, `integer_eval`), with
  `LIBC_MATH_<FUNC>_IMPL` selecting the one to use. This is the intended
  structure for every function, although only some functions have been
  converted so far. Variants other than the default may make different
  guarantees.
- The functions in `math::static_rounding` take the rounding mode as an
  argument. They exist for callers that need a fixed rounding mode, such as
  compilers. They are not the primary implementation of a function. The
  primary implementation is whichever one is best for the target's
  configuration.
- A pull request may add a function for a single floating point type. It does
  not need to cover every type at once.

### Testing

- The author of a math change runs the unit, smoke, and exhaustive tests for
  it on their own machine, at least in the default configuration, and
  preferably before sending the pull request. Reviewers check that this was
  done. CI covers most of the tests and some configurations, not all of them.
- The exhaustive tests take a long time and are not run by `check-libc`.
- If the tests are hard to run, or hard to get passing, ask the math
  maintainers for help.
- Tests that compare against MPFR run only in overlay mode. Every function
  also needs a smoke test, which is what runs in full builds and on baremetal
  and GPU targets.
- `libc/test/src/math/exhaustive/` contains hard-to-round inputs from the
  [CORE-MATH](https://core-math.gitlabpages.inria.fr/) project, as text files
  (`.wc`) and in a compact binary form (`.bin`). The two forms must stay
  consistent.
