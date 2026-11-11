(code_organization)=

# Where Code Goes

This page explains how to decide where a piece of code belongs: in an
entrypoint, in a helper next to it, in `src/__support/`, or in `shared/`. It
also covers which build modes and targets a new function is enabled for.
{ref}`source_tree_layout` describes the directories themselves.

## Three kinds of code

It helps to sort code in a libc into three kinds:

- **OS independent**: pure computation with no system interaction, such as
  the math and string functions.
- **OS agnostic**: standard interfaces whose behavior is the same everywhere
  but whose implementation needs the system, such as `fopen` or `assert`.
- **OS specific**: interfaces that are specific to a platform. This can be a
  thin interface like a syscall wrapper or a substantial integration like
  `popen`.

The source tree does not separate these three as cleanly as it should, and
moving toward that separation is a long-term goal. When adding code, decide
which kind it is and keep OS-specific details out of the first two.

## Entrypoints do not call entrypoints

An entrypoint is the definition of one public function. It may contain its
whole implementation. As soon as any part of that implementation is needed
somewhere else, that part moves into a separate internal header or library.

**An entrypoint must not call another entrypoint.** This is the most common
mistake in new code. Calling `strlen` from `strdup`, or `fseek` from
`fsetpos`, is the natural way to write the function. It is wrong here because
every entrypoint is a separate public symbol in its own separate object file
that each target enables or leaves out individually. A dependency between two
entrypoints means that they must be pulled in together.

Instead, put the shared logic in an internal function and call it from both
entrypoints:

```c++
// src/string/string_length.h
namespace LIBC_NAMESPACE_DECL {
namespace internal {
LIBC_INLINE size_t string_length(const char *src) { ... }
} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

// src/string/strlen.cpp
LLVM_LIBC_FUNCTION(size_t, strlen, (const char *src)) {
  return internal::string_length(src);
}
```

CMake checks for this. When configured with `-DLIBC_CMAKE_VERBOSE_LOGGING=ON`,
it warns about any entrypoint or internal library that depends on an
entrypoint. The few permitted exceptions, currently `errno` and `longjmp`, are
listed in `check_entrypoint_deps` in
`libc/cmake/modules/LLVMLibCObjectRules.cmake`.

Internal libraries may depend on *variable* entrypoints such as `environ` or
`stdout` when they need the object itself. They may not depend on function
entrypoints.

## Where shared logic lives

| The logic is used by | Put it in |
| --- | --- |
| One entrypoint | The entrypoint's `.cpp` file, as `static` functions. |
| A few entrypoints, normally from the same public header | A header or object library in that header's directory, for example `src/stdio/fflush_internal.h`, in the `internal` namespace. |
| Many parts of the library, or other LLVM projects | `src/__support/`. |

Avoid creating new files or directories named `utils`. A name that describes
nothing attracts unrelated code. Name a helper after what it does.

Keep internal interfaces as narrow and as typed as public ones:

- Functions whose result becomes `errno` return `ErrorOr<T>` and never set
  `errno` themselves. See {ref}`setting_errno`.
- Reuse an existing parser or type before writing a narrower one. For example,
  `stdio` mode strings are parsed by `FileMode`.
- Write validity checks positively ("is one of the supported values"), not as
  a list of rejected cases.
- Prefer a fixed signature over an unconstrained template when the body only
  works for specific types.

## Code shared with other LLVM projects

Headers in `libc/shared/` expose LLVM-libc internals to other LLVM projects
through {ref}`Hand-in-Hand <hand_in_hand>`.

`shared/` is only an interface. The implementation lives under `libc/src/`,
usually in `src/__support/`, and the header in `shared/` re-exports it in the
`shared` namespace. Code inside LLVM-libc includes the implementation from
`src/` and does not include anything from `shared/`. The GPU RPC headers are
an existing exception: they are implemented in `shared/` itself.

Everything reachable from `shared/` has extra constraints:

- It is header-only.
- It does not depend on any external libc function. The allocator is the one
  exception, because it is external to LLVM-libc on most targets already.
- It builds with the compilers LLVM itself supports, which are older than the
  minimum for the rest of LLVM-libc. See {ref}`compiler_support`.

This covers the `printf` core, string-to-number conversion, the math
implementations in `src/__support/math/`, and what they include from
`src/__support/`. Adding a new header to `shared/`, or a new consumer of it,
should be discussed with the maintainers first.

## Operating-system-specific code

Per-OS duplication is acceptable. Operating systems differ enough that sharing
an implementation between them usually costs more than it saves.

System calls are made only from wrappers in `src/__support/OSUtil/<os>/`; see
{ref}`syscall_wrapper_refactor`. OS-agnostic code never names an OS-specific
namespace such as `linux_syscalls`.

There are several ways to select the implementation for the target OS. Pick the
one that fits the amount of code that differs:

- **The whole entrypoint differs.** Put each implementation in
  `src/<header>/<os>/<func>.cpp`, and make the entrypoint in
  `src/<header>/CMakeLists.txt` an `ALIAS` that depends on
  `.${LIBC_TARGET_OS}.<func>`. Most syscall-backed functions work this way.
- **The entrypoint is common and one step differs.** Keep a single
  `src/<header>/<func>.cpp` that calls an `internal::` function, and provide
  that function as a per-OS object library selected with the same `ALIAS`
  mechanism. `fflush` and its `fflush_internal` library are the example.
- **A small header-only primitive differs.** Use a dispatch header that
  includes the right implementation, as `src/__support/OSUtil/io.h` does.

## Overlay mode and full build

A function may be enabled in {ref}`overlay mode <overlay_mode>` only if it
does not depend on an implementation-defined type or on library-owned state
that is shared between functions. For example:

- `strlen` and `round` are pure functions, so they are fine in overlay mode.
- `fopen` and `fgetc` are full build only, because `FILE` is implementation
  defined.
- `atexit` is full build only, because it sets state that `exit` must read.
- `strtok` is fine in overlay mode. It keeps internal state, but no other
  function reads that state.

In each `entrypoints.txt`, overlay-safe functions are listed unconditionally and
the others are inside `if(LLVM_LIBC_FULL_BUILD)`.

Both modes matter. Overlay mode is the more portable of the two and is how many
users adopt the math and memory functions; a full build is where everything
else is exercised. A change must build and pass its tests in both modes unless
the code is full-build only. See {ref}`testing_guide`.

## Which targets get a new function

A new entrypoint should be enabled on every target where it works. In practice
contributors cannot test every target, so the minimum is:

- For a function that needs Linux, the `entrypoints.txt` files for `x86_64`,
  `aarch64`, `arm`, and `riscv` under `libc/config/linux/`. The Linux
  configuration for 32-bit Arm is also how that architecture is tested; see
  {ref}`platform_support`.
- For an OS-independent function, the Linux configurations above plus the
  baremetal, GPU, and other OS configurations, as described for math functions
  in `libc/src/math/docs/add_math_function.md`.

`linux/x86_64` is the most complete configuration. An entrypoint that is
present there and missing from another Linux architecture is usually an
oversight, not a decision, and adding it is a welcome change.

The public header declaration comes from `libc/include/<header>.yaml`. A
function that is missing there will build and pass the tests but will not appear
in the generated header.

### Experimental entrypoints

Entrypoints with known-incomplete implementations are listed inside
`if(LLVM_LIBC_ENABLE_EXPERIMENTAL_ENTRYPOINTS)` and are off by default. See
{ref}`extensions_and_stubs` for when a function belongs there.

## Configuration options

Options in `libc/config/config.json` are named `LIBC_CONF_*` and are CMake
settings. They become `LIBC_COPT_*` macros on the compiler command line. See
{ref}`configure_options`.

Add an option when a standard requires one behavior and a real user needs
another. Disabling floating point support in `printf` for small targets is the
model case. Do not add an option to avoid making a design decision.

An option that disables behavior in an entrypoint must disable the matching
checks in that entrypoint's tests under the same condition.

## Bazel

The Bazel build in `utils/bazel/llvm-project-overlay/libc/` is maintained by
its downstream users, and an automated process keeps it in sync. Upstream
contributors are not required to update it, but it is appreciated.
