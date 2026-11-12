(testing_guide)=

# Testing Guide

This page describes how LLVM-libc tests are organized, what a test for a new
function should cover, and how to make sure the test you wrote is the test
that ran. {ref}`build_and_test` lists the build commands.

## Kinds of tests

All tests are declared with `add_libc_test` or one of its wrappers, in the
`CMakeLists.txt` next to the test source under `libc/test/`.

| Rule | Use it for |
| --- | --- |
| `add_libc_test` | The default. The same source is built against the system libc in overlay mode and against LLVM-libc's own startup code and headers in a full build. |
| `add_fp_unittest` | Floating point tests. Adds the floating point test helpers. |
| `add_integration_test` | Behavior that needs a real process from start to finish: program startup, `exec`, thread-local storage, threads still running at exit. Full build only. |

A test may be restricted with `OVERLAY_ONLY` or `FULL_BUILD_ONLY`. Use these
only when the test cannot work in the other mode. An entrypoint that is
full-build only does not need the option: its test is skipped automatically in
overlay mode because the entrypoint is missing.

Each test names the `SUITE` it belongs to, and each test directory defines a
suite named after it, such as `libc-ctype-tests`.

## Running tests

```sh
# Build and run one test.
ninja -C build libc.test.src.ctype.isalpha_test

# Build and run every test in one directory.
ninja -C build libc-ctype-tests

# Build and run everything.
ninja -C build check-libc
```

While developing, run the specific test. Before considering a change finished,
run `check-libc`, because a change in one area sometimes affects another.

A change should pass in both an overlay build and a full build. Keeping one
build directory for each mode makes this quick.

## Making sure your test ran

CMake silently drops a test when one of its dependencies is not available on
the target: an entrypoint that is not in `entrypoints.txt`, a missing startup
object, or a missing MPFR library. A dropped test does not fail. It is simply
not part of `check-libc`, which then passes. This is expected behavior, since
some entrypoints are unavailable on some targets and should not report an error.

To avoid reporting a pass for a test that never ran:

- Configure with `-DLIBC_CMAKE_VERBOSE_LOGGING=ON`. CMake then prints a line
  starting with `Skipping` for every test dropped because of a missing
  entrypoint or startup object. Tests dropped because MPFR is missing are
  reported only at CMake's verbose log level (`--log-level=VERBOSE`).
- Run the test by its own target name. If the target does not exist, the test
  was dropped.
- Watch it fail first. Run the test before the implementation is complete, or
  break the implementation on purpose, and confirm that the test fails.

## What to test

Test the behavior that LLVM-libc is responsible for.

- **A function that is a thin wrapper around a system call** needs one test of
  success and one of failure. That shows the arguments reach the kernel in the
  right order and that an error is reported through the return value and
  `errno`. Do not test path resolution, permission checks, or other behavior
  the kernel implements.
- **Logic that lives in the libc** is tested thoroughly, including the boundary
  cases. A search of `PATH` in `execvp` and the buffering in `stdio` are
  examples.
- **Tests for C standard functions use only C standard functions** to
  set up and check their state. A test for `fgets` should not need `pipe` or
  `fdopen`.

A test file is generally named after the function it tests. When one file
tests several functions that only make sense together, name it after all of
them, for example `popen_pclose_test.cpp`. This is a general rule and not an
absolute one: `fileop_test.cpp` covers many `stdio` functions under one
descriptive name.

### Conventions

- Test suite names start with `LlvmLibc`.
- Call the LLVM-libc functions through the `LIBC_NAMESPACE` namespace to avoid
  pulling in the system libc's implementations.
- Include types and macros through the proxy headers in `libc/hdr/`, as in
  implementation code.
- For functions that set `errno`, derive the fixture from `ErrnoCheckingTest`
  and check results with the `Succeeds` and `Fails` matchers from
  `test/UnitTest/ErrnoSetterMatcher.h`. The fixture asserts that `errno` is
  zero when each test ends, so a test that expects an error must consume it.
- Compare with `ASSERT_EQ`, `ASSERT_NE`, and the other typed assertions, not
  `ASSERT_TRUE(a == b)`.
- Use `cpp::scope_exit` to close files and remove temporary files when a later
  assertion could return early.

## Tests must not depend on their environment

Tests run on developers' machines, in containers, under emulators, and on
buildbots, as different users.

- **Paths.** Create every path with `libc_make_test_file_path`, including
  paths that are meant not to exist. Never assume that a fixed absolute path
  such as `/nonexistent` is absent.
- **Privileges.** Avoid tests whose result depends on the user's privileges.
  Where that is unavoidable, the test must pass both as root and as an
  ordinary user.
- **System state.** Do not rely on `/proc` being mounted or on network access.
- **Emulators.** A test that cannot work under user-mode emulation is guarded
  with `LIBC_TEST_UNDER_EMULATOR` instead of having its expectations loosened.

## Disabling a test

A test that cannot pass on some target is disabled in CMake, with a condition
that names the target and a link to an issue that tracks re-enabling it. Do
not comment out a test body or wrap it in `#if 0`.

## Death tests and null checks

`EXPECT_DEATH` and `EXPECT_EXITS` tests, including the tests that check
`LIBC_CRASH_ON_NULLPTR`, are wanted. Some CI configurations currently skip
them with `LIBC_TEST_SKIP_DEATH_TESTS`; that is a limitation of those
configurations to be fixed and not a reason to leave such tests out.

## Other test frameworks

Tests are written against LLVM-libc's own framework in `libc/test/UnitTest/`,
which is a subset of GoogleTest. Some downstream users compile the same tests
with GoogleTest or a compatible framework, so stay within that subset and the
matchers LLVM-libc provides. Keeping a downstream framework working is the
responsibility of that downstream.

## Math tests

- Tests that compare against MPFR (`NEED_MPFR`) run in overlay mode only,
  because MPFR is not yet supported in a full build. A math change must be
  tested in an overlay build with MPFR installed.
- Tests in `libc/test/src/math/smoke/` do not need MPFR. They are the math
  tests that run in full builds and on baremetal and GPU targets.
- Tests in `libc/test/src/math/exhaustive/` take a long time and are not run
  by `check-libc`. The author of a math change runs them locally. See
  {ref}`math_function_guidelines`.

## Fuzzing and benchmarks

Neither is required for a new function today. Ideally, security-sensitive
functions have fuzz tests and performance-sensitive functions have benchmarks.

Fuzz tests live in `libc/fuzzing/` (see {ref}`fuzzing`). A migration of the
fuzz tests to [FuzzTest](https://github.com/google/fuzztest) is being
considered. The benchmarks under
`libc/benchmarks/` are unmaintained, apart from the GPU benchmarks in
`libc/benchmarks/gpu/`.
