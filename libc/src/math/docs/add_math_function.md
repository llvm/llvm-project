# How to add a new math function to LLVM-libc

This document is to serve as a cookbook for adding a new math function
implementation to LLVM libc.  To add a new function, apart from the actual
implementation, one has to follow a few other steps to setup proper registration
and shipping of the new function.  Each of these steps will be described in
detail below.

## Registration

To register the function's entry points for supported OSes and architectures,
together with its specifications:

- Add entry points `libc.src.math.func` to the following files:
```
  libc/config/linux/<arch>/entrypoints.txt
  libc/config/windows/entrypoints.txt
```
- Add function specs to the file:
```
  libc/include/math.yaml
```

## Implementation

The function's actual implementation is defined in an internal header, while the public entry point is a thin wrapper 
added to the following locations:

- Add the core math logic (under `LIBC_NAMESPACE::math` namespace) to:
```
  libc/src/__support/math/<func>.h
```
- Add the corresponding `add_header_library` to:
```
  libc/src/__support/math/CMakeLists.txt
```
- Add `add_math_entrypoint_object(<func>)` to:
```
  libc/src/math/CMakeLists.txt
```
- Add function declaration (under `LIBC_NAMESPACE` namespace) to:
```
  libc/src/math/<func>.h
```
- Add function definition (calling the __support implementation) to:
```
  libc/src/math/generic/<func>.cpp
```
- Add the corresponding `add_entrypoint_object` to:
```
  libc/src/math/generic/CMakeLists.txt
```
- Add architectural specific implementations to:
```
  libc/src/math/<arch>/<func>.cpp
```

### Shared Math Library

If the function should be available to internal LLVM projects:

- Include the support header in:
```
  libc/shared/math.h
```
- Add a header that exports the function via using in:
```
  libc/shared/math/<func>.h
```
- add a simple test case to
```
 libc/test/shared/shared_math_test.cpp
```
- add the corresponding `libc_support_library` and `libc_math_function` to:
```
utils/bazel/llvm-project-overlay/libc/BUILD.bazel
```

### Floating point utility

- Floating point utilities and math functions that are also used internally are
located at:
```
  libc/src/__support/FPUtils
```
- These are preferred to be included as header-only.
- To manipulate bits of floating point numbers, use the template class
`LIBC_NAMESPACE::fputil::FPBits<>` in the header file:
```
  libc/src/__support/FPUtils/FPBits.h
```

## Testing

### MPFR utility

In addition to the normal testing macros such as `EXPECT_EQ, ASSERT_THAT, ...`
there are two special macros `ASSERT_MPFR_MATCH` and `EXPECT_MPFR_MATCH` to
compare your outputs with the corresponding MPFR function.  In
order for your new function to be supported by these two macros,
the following files will need to be updated:

- Add the function enum to `LIBC_NAMESPACE::testing::mpfr::Operation` in the
header file:
```
  libc/utils/MPFRWrapper/MPFRUtils.h
```
- Add support for `func` in the `MPFRNumber` class and the corresponding link
between the enum and its call to the file:
```
  libc/utils/MPFRWrapper/MPFRUtils.cpp
```

### Unit tests

Besides the usual testing macros like `EXPECT_EQ, ASSERT_TRUE, ...` there are
testing macros specifically used for floating point values, such as
`EXPECT_FP_EQ, ASSERT_FP_LE, ...`

- Add smoke tests (simple cases and zeros / inf / nan inputs or outputs) to:
```
  libc/test/src/math/smoke/<func>_test.cpp
```
- Add unit test that might require MPFR to:
```
  libc/test/src/math/<func>_test.cpp
```
- Add the corresponding entry points to:
```
  libc/test/src/math/smoke/CMakeLists.txt
  libc/test/src/math/CMakeLists.txt
```

### Exhaustive tests

Exhaustive tests are long-running tests that are not included when you run
`ninja check-libc`.  These exhaustive tests are added and manually run in
order to find exceptional cases for your function's implementation.

- Add an exhaustive test to:
```
  libc/test/src/math/exhaustive/<func>_test.cpp
```
- Add the corresponding entry point to:
```
  libc/test/src/math/exhaustive/CMakeLists.txt
```
- The template class `LlvmLibcExhaustiveMathTest` located at:
```
  libc/test/src/math/exhaustive/exhaustive_test.h
```
can be used for conveniently parallelizing the exhaustive tests.

### Performance tests

Performance tests compare your function's implementation with the system libc
implementation (which is very often glibc).

- Add a performance test to:
```
  libc/test/src/math/performance_testing/<func>_perf.cpp
```
- Add the corresponding entry point to:
```
  libc/test/src/math/performance_testing/CMakeLists.txt
```

## Build and Run

- Check out the LLVM source tree:
```
  $ git clone https://github.com/llvm/llvm-project.git
```

- Compiler Requirements: The libc implementation should support compiling with both Clang and GCC (12.2).

- Setup projects with CMake:
```
  $ cd llvm-project
  $ mkdir build
  $ cd build
  $ cmake ../llvm -G Ninja \
  -DLLVM_ENABLE_RUNTIMES="libc" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++
  $ cd runtimes/runtimes-bins/
```

- Build the whole `libc`:
```
  $ ninja libc
```

- Run all unit tests:
```
  $ ninja check-libc
```

- Run math smoke tests only:
```
  $ ninja libc-math-smoke-tests
```

- Run math smoke and unit tests:
```
  $ ninja libc-math-unittests
```

- Build and Run a specific unit test:
```
  $ ninja libc.test.src.math.<func>_test.__unit__
  $ projects/libc/test/src/math/libc.test.src.math.<func>_test
```

- Build and Run exhaustive test (might take hours to run):
```
  $ ninja libc.test.src.math.exhaustive.<func>_test.__unit__
  $ projects/libc/test/src/math/exhaustive/libc.test.src.math.exhaustive.<func>_test.__unit__
```

- Build and Run performance test:
```
  $ ninja libc.test.src.math.performance_testing.<func>_perf
  $ projects/libc/test/src/math/performance_testing/libc.test.src.math.performance_testing.<func>_perf
  $ cat <func>_perf.log
```

## Code reviews

We use GitHub's inbuilt pull request system for code review:
```
  https://docs.github.com/articles/about-collaborative-development-models
  https://docs.github.com/articles/about-pull-requests
```
