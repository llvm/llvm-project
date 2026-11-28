(roadmap)=

# Roadmap and Known Issues

This page collects planned features and known problems in LLVM-libc. It is a
working list, and each numbered item is intended to become an issue in the
[issue tracker](https://github.com/llvm/llvm-project/labels/libc). Check the
tracker, or ask the maintainers, before starting on one.

Paths are relative to `libc/` unless they start with `.github/` or `.ci/`.
Counts and line numbers are as of November 2026.

## Planned features

These are missing or incomplete today. They are not non-goals; see
{ref}`design_philosophy` for those.

| Feature | Status |
| --- | --- |
| Dynamic loading | Under development. Never intended to be the primary use case. |
| Time zone database | Planned. The intent is to base the implementation on the one in libc++. |
| Thread cancellation | Low priority. When added, it must not significantly slow down code that does not use it. |
| Message catalogs, Annex K | Low priority. |
| Locales | The `_l` variants of the relevant functions are planned. See {ref}`design_philosophy`. |
| UTF-16 `wchar_t` | In progress. |
| Full build for macOS and Windows | Not designed yet. See {ref}`abi_policy`. |

## Code cleanup
// TODO(michaelrj): This section should probably be converted into github
// issues, but I'm leaving it here for now to make it easier to review.
// Plan is to remove this section before merging.

The areas in this section do not follow the current guidelines. Do not use
them as models for new code.

### 1. Unify the `File` abstraction across Linux, baremetal, and GPU

- The Linux implementation is in `src/__support/File/`. The baremetal and GPU
  implementations are in `src/stdio/baremetal/` and `src/stdio/gpu/`.
- Goal: one design shared by all three.

### 2. Clean up `src/__support/HashTable`

- Three parts: implementation style, header inclusion, and
  architecture-specific dispatch.
- Dispatch today: `HashTable/bitmask.h` includes `sse2/bitmask_impl.inc` or
  `generic/bitmask_impl.inc` depending on `LIBC_TARGET_CPU_HAS_SSE2` and
  `__LIBC_EXPLICIT_SIMD_OPT`.
- `HashTable/randomness.h:27` and `:30` use raw `thread_local` instead of
  `LIBC_THREAD_LOCAL`.

### 3. Layer the string-to-float parsers and fallbacks more clearly

- `src/__support/str_to_float.h`. The original string pointer has to be passed
  along even after the first parser has consumed the input.

### 4. Rename `src/__support/StringUtil`

- It holds only the `strerror` and `strsignal` text
  (`error_to_string.*`, `signal_to_string.*`, `message_mapper.h`, tables).

### 5. Separate OS-independent, OS-agnostic, and OS-specific code

- The long-term direction described in {ref}`code_organization`. This is
  probably a series of changes more than a single issue.

### 6. Remove entrypoint-to-entrypoint calls

- `src/signal/linux/signal.cpp:26` calls `LIBC_NAMESPACE::sigaction`.
- `src/pthread/pthread_attr_getstack.cpp:24` calls
  `LIBC_NAMESPACE::pthread_attr_getstacksize`.
- These two were found by inspection, and there are likely more. The CMake
  check (`check_entrypoint_deps`) only warns under
  `LIBC_CMAKE_VERBOSE_LOGGING`.
- See {ref}`entrypoints_do_not_call_entrypoints`.

### 7. Replace `<>` includes of public headers under `libc/src` with proxy headers

- `#include <pthread.h>` appears in 119 files under `src/`.
- `#include <inttypes.h>` appears in 8 files under
  `src/__support/printf_core/`, which is also built in overlay mode.

### 8. Include `src/__support/CPP` headers through the top-level header

- Files outside `src/__support/CPP/` that include a sub-header directly:
  6 for `type_traits/`, 6 for `utility/`, 3 for `cstddef/`.

### 9. Finish the syscall wrapper migration

- 24 files under `src/` outside `src/__support/OSUtil` still mention
  `syscall_impl`.
- `src/__support/OSUtil/linux/syscall_wrappers/open.h` uses `syscall_impl`
  with a hand-written sign check instead of `syscall_checked`.
- See {ref}`syscall_wrapper_refactor`.

### 10. Move incomplete entrypoints behind the experimental flag

- In `config/linux/x86_64/entrypoints.txt`, `libc.src.unistd.sysconf` and
  `libc.src.unistd.confstr` are in the default full-build list, and
  `libc.src.dlfcn.dlopen` (returns `nullptr`) is in the overlay list. The
  `LLVM_LIBC_ENABLE_EXPERIMENTAL_ENTRYPOINTS` block starts at line 1801.
- See {ref}`extensions_and_stubs`.

### 11. Sync the `linux/aarch64` entrypoint list with `linux/x86_64`

- 227 entrypoints are in `config/linux/x86_64/entrypoints.txt` and not in
  `config/linux/aarch64/entrypoints.txt`.

## Public headers and ABI

### 12. Make `alloca` available from `<stdlib.h>`

- `include/alloca.yaml` exists. `include/stdlib.yaml` does not reference it.

### 13. Fix reserved-name and portability problems in `llvm-libc-types`

See {ref}`writing_public_headers` for the rules.

- `__barrier_type.h`: members `expected`, `waiting`, `mutex`; includes
  `<stdbool.h>`.
- `locale_t.h`: defines the macro `NUM_LOCALE_CATEGORIES`.
- `struct_epoll_event.h`: uses `[[gnu::packed]]`.
- `__jmp_buf.h`: members such as `rbx` and `opaque`.
- `imaxdiv_t.h`, `struct_dl_phdr_info.h`, `struct_flock.h`,
  `struct_flock64.h`: include `<stdint.h>`.
- `pthread_barrierattr_t.h`: includes `<stdbool.h>`.
- `struct_termios.h`: the include guard
  `__LLVM_LIBC_TYPES_STRUCT_TERMIOS_H__` differs from every other type header.

### 14. Remove the `_LARGEFILE64_SOURCE` guard

- `include/llvm-libc-macros/linux/fcntl-macros.h:110`. It is the one
  feature-test macro in the public headers that no standard requires. See
  {ref}`feature_test_macros`.

### 15. Make `ino_t`, `nlink_t`, and `blkcnt_t` 64-bit on every target

- `ino_t` and `nlink_t` are `__UINTPTR_TYPE__`; `blkcnt_t` is
  `__INTPTR_TYPE__`. `off_t`, `time_t`, `dev_t`, and `rlim_t` are 64-bit.

### 16. Decide a minimum kernel-headers version

- `config/linux/x86_64/exclude.txt` probes the host for headers and syscall
  numbers instead.

## Math

### 17. Make double precision `atan` and `atan2` correctly rounded

- `utils/docgen/math.yaml:995` (`atan`, double: 1 ULP), `:1010` (`atan2`,
  double: 1 ULP), `:1011` (`atan2`, float128: 1 ULP).
- A fix is in progress.

### 18. Keep the exhaustive test data files consistent, and document adding cases

- `test/src/math/exhaustive/` has `sin.wc`, `cos.wc` (text) and `sin.bin`,
  `cos.bin` (binary) with hard-to-round cases from CORE-MATH.
- Planned: a CI check that the two forms are consistent, and instructions for
  adding more cases.

### 19. Document what each math implementation variant guarantees

- Variants: `double_eval`, `float_eval`, `float_eval_cr_rn`, `integer_eval`,
  selected by `LIBC_MATH_<FUNC>_IMPL`.
- Planned: a table on the {ref}`math page <math>` once more variants exist.

### 20. Set expectations for testing math changes across floating point configurations

- Baremetal users see many math test failures downstream, because the tests
  assume a 64-bit target with complete hardware floating point support.
- To decide: what testing against different floating point and bit-width
  configurations is expected of a math change.

### 21. Make MPFR tests work in a full build

- `NEED_MPFR` forces a test to be overlay-only in
  `cmake/modules/LLVMLibCTestRules.cmake`.

### 22. Re-enable the `#if 0` blocks in the shared math constexpr test

- `test/shared/shared_math_constexpr_test.cpp:221` and `:312`. The second
  cites issue #185232.

## Testing and CI

### 23. Re-enable death tests in presubmit

- `-DLIBC_TEST_SKIP_DEATH_TESTS=ON` is set in `.ci/monolithic-linux.sh:47` and
  `.github/workflows/libc-fullbuild-tests.yml:209`.

### 24. Compile the tests with `-Werror`

- `cmake/modules/LLVMLibCTestRules.cmake:96` has the `-Werror` line commented
  out, because the tests do not all build cleanly with it. The library code
  is compiled with `-Werror`.

### 25. Restore the AArch64 and RISC-V 64 postcommit builders

- Postcommit currently covers Linux on x86-64 only (Clang, GCC, full build,
  overlay, ASan). The AArch64 Linux builder needs fixing and the RISC-V 64
  builder needs a new setup.

### 26. Settle and document the minimum compiler versions

- `docs/compiler_support.md` says Clang 11 / GCC 12.2.
  `docs/getting_started.md` says Clang 15+, `docs/full_host_build.md` says
  Clang 21 was tested, and `docs/complete_runtimes_build.md` says at least
  Clang 22.
- The answer depends on what the postcommit builders run.

### 27. Decide what to do with `libc/benchmarks`

- It is unmaintained apart from `benchmarks/gpu/`, and will be removed or
  rewritten. A restructuring that follows the unification of the CMake test
  rules has been discussed.

### 28. Improve fuzzing coverage and infrastructure

- `libc/fuzzing/` is in use but covers less than it should. The fuzzers are
  built in one presubmit configuration and are not run there.
- A migration to [FuzzTest](https://github.com/google/fuzztest) is being
  considered.
- Building the fuzzers with coverage instrumentation (sancov) by default has
  been suggested. Today `fuzzing/CMakeLists.txt` adds `-fsanitize=fuzzer` for
  that directory only.

## Documentation

### 29. Update the allocation section of the code style page

- {ref}`code_style` ("Allocations in the libc runtime code") documents only
  `AllocChecker`. `src/__support/checked_ptr.h` (`CheckedPtr`, used through
  `make_unique`) is newer.
- To decide: whether `CheckedPtr` replaces `AllocChecker`.

### 30. Fix stale pages

- `docs/gpu/building.md`, `docs/gpu/testing.md`, and `docs/gpu/using.md` name
  `amdhsa-loader`, `nvptx-loader`, and RPC server artifacts that are not in
  the tree.
- `docs/dev/header_generation.md` shows usage for `yaml_to_classes.py` with a
  `--e` flag and presents `.h.def` as the main mechanism.
- `docs/dev/code_style.md` says `libc/src/.clang-tidy` enables the
  `llvmlibc-*` checks. It enables one of them.
- `docs/complete_runtimes_build.md` tells readers to check out a demo pull
  request.
- `docs/math/index.md` "Source Locations" omits `src/__support/math` and
  `shared/`.

## Open questions

- May a new default math implementation land before it is correctly rounded,
  and if so, with what documented bound?
  {ref}`math_function_guidelines` lists only the exceptions that exist today.

## Documentation to update when an item lands

| When this lands | Update |
| --- | --- |
| Item 9 | {ref}`syscall_wrapper_refactor`: "The migration of existing code ... is still in progress" |
| Item 10 | {ref}`extensions_and_stubs`: "Some incomplete entrypoints predate the option" |
| Item 15 | {ref}`abi_policy`: "A few related types ... are still pointer width" |
| Item 21 | {ref}`testing_guide` and {ref}`math_function_guidelines`: MPFR tests are overlay-only |
| Item 23 | {ref}`testing_guide`: "Some CI configurations currently skip them" |
| Item 24 | {ref}`porting`: "The tests are currently compiled without `-Werror`" |
| Item 25 | {doc}`arch_support`: the sentence on postcommit coverage |
| Items 27, 28 | {ref}`testing_guide`: "Fuzzing and benchmarks" |
| More baremetal targets tested in presubmit | {doc}`arch_support` table, {ref}`platform_support` |
