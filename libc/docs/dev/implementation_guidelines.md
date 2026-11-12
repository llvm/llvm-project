(implementation_guidelines)=

# Implementation Guidelines

{ref}`code_style` covers naming, macros, and formatting. This page covers the
judgment calls that come up while writing an implementation: what to optimize
for, how to use memory, when to check inputs, and which parts of the tree are
known to need rework.

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
benchmarking setup today.

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
already prefer. On baremetal targets LLVM-libc provides an optimized allocator
in `src/__support/freelist_heap.h`.

## Runtime checks

- **Null pointers.** At the start of an entrypoint, pass each pointer argument
  that would be dereferenced, and for which a null pointer is not a valid
  input, to `LIBC_CRASH_ON_NULLPTR` (`src/__support/macros/null_check.h`). The
  check is lightweight, is controlled by the `LIBC_ADD_NULL_CHECKS` option, and
  exists to make a crash easy to diagnose. Put it in the entrypoint only, not
  in internal code.
- **Internal invariants.** Use `LIBC_ASSERT`. It compiles away in release
  builds.
- **Hardening.** `LIBC_COPT_HARDENING_MODE` was added for the allocator's
  integrity checks. Extending it to other code needs discussion first.
- **Everything else.** Do not validate inputs beyond what the standard
  requires. {ref}`undefined_behavior` explains the reasoning.

## Headers

- **Include what you use.** A `.cpp` file that names `mode_t` includes
  `hdr/types/mode_t.h` itself and does not rely on another header to provide
  it.
- **Use proxy headers** from `libc/hdr/` for every type and macro that the
  libc also provides publicly. See the header inclusion policy in
  {ref}`code_style`. Compiler-provided headers are tolerated but should also
  go through a proxy header where one exists. The existing `#include
  <pthread.h>` style includes under `libc/src/` are incorrect and should not
  be copied.
- **Type traits** come from the umbrella header
  `src/__support/CPP/type_traits.h`, not from the individual files under
  `src/__support/CPP/type_traits/`.

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

## Math functions

```{note}
This section records the general expectations as understood by the
non-math maintainers. It needs review by the math maintainers.
```

- New functions are expected to be correctly rounded in all rounding modes.
  The {doc}`math.h status tables <../headers/math>` record the accuracy of
  each function.
- The implementation lives in a header under `src/__support/math/`, and the
  entrypoint is a thin wrapper. `libc/src/math/docs/add_math_function.md` is
  the step-by-step guide.
- A pull request may add a function for a single floating point type. It does
  not need to cover every type at once.
- New code converts a signaling NaN input to a quiet NaN and raises
  `FE_INVALID`.
- `errno` and floating point exceptions are reported only through
  `fputil::set_errno_if_required` and `fputil::raise_except_if_required`.
- Tests that compare against MPFR run only in overlay mode. Every function
  also needs a smoke test, which is what runs in full builds and on baremetal
  and GPU targets.
- The GPU-specific math implementations have different accuracy bounds and
  are maintained by the GPU maintainers.

```{admonition} Open questions for the math maintainers
:class: important

- Is the multiple-implementation scheme (`LIBC_MATH_<FUNC>_IMPL` selecting
  among `double_eval`, `float_eval`, and `integer_eval` variants) the intended
  structure for every function? What does each variant promise for accuracy,
  `errno`, exceptions, and signaling NaNs?
- Is `math::static_rounding` the future primary implementation, with the
  dynamic rounding mode as a wrapper, or a parallel API for compiler use only?
- Must a new single precision or half precision function pass an exhaustive
  test in all four rounding modes before it is merged, and who runs it?
- The status tables list double precision `atan` and `atan2` at 1 ULP. Is
  that an accepted bound or a gap to close?
- What are the rules for the test data under
  `libc/test/src/math/exhaustive/`?
```

## Areas that need rework

The following parts of the tree are known to fall short of the guidelines on
this page. Do not use them as models for new code. Improvements are welcome
as separate, focused changes.

- **The `File` abstraction.** The Linux implementation is in
  `src/__support/File/`, while the baremetal and GPU implementations are in
  `src/stdio/baremetal/` and `src/stdio/gpu/`. These should share one design.
- **`src/__support/HashTable/`** needs a cleanup.
- **String to floating point conversion.** The layering between the parsers
  and their fallbacks should be clearer. Today the original string has to be
  passed along even after the first parser has consumed it.
- **`src/__support/StringUtil/`** holds only the `strerror` and `strsignal`
  text and should be renamed for what it contains.
- **Separation by OS dependence.** See {ref}`code_organization`.
- **`libc/benchmarks/`** is unmaintained, apart from the GPU benchmarks in
  `libc/benchmarks/gpu/`, and will be removed or rewritten.
- **`libc/fuzzing/`** is in use but covers less than it should.
