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
  libc/spec/stdc.td
```

## Implementation

The function's actual implementation and its corresponding header should be
added to the following locations:

- Add `add_math_entrypoint_object(<func>)` to:
```
  libc/src/math/CMakeLists.txt
```
- Add function declaration (under `__llvm_libc` namespace) to:
```
  libc/src/math/<func>.h
```
- Add function definition to:
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

### Floating point utility

- Floating point utilities and math functions that are also used internally are
located at:
```
  libc/src/__support/FPUtils
```
- These are preferred to be included as header-only.
- To manipulate bits of floating point numbers, use the template class
`__llvm_libc::fputil::FPBits<>` in the header file:
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

- Add the function enum to `__llvm_libc::testing::mpfr::Operation` in the
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
  libc/test/src/math/differential_testing/<func>_perf.cpp
```
- Add the corresponding entry point to:
```
  libc/test/src/math/differential_testing/CMakeLists.txt
```

## Build and Run

- Check out the LLVM source tree:
```
  $ git clone https://github.com/llvm/llvm-project.git
```

- Setup projects with CMake:
```
  $ cd llvm-project
  $ mkdir build
  $ cd build
  $ cmake ../llvm -G Ninja \
  -DLLVM_ENABLE_PROJECTS="llvm;libc" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++
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
  $ ninja libc.test.src.math.<func>_test
  $ projects/libc/test/src/math/libc.test.src.math.<func>_test
```

- Build and Run exhaustive test (might take hours to run):
```
  $ ninja libc.test.src.math.exhaustive.<func>_test
  $ projects/libc/test/src/math/exhaustive/libc.test.src.math.exhaustive.<func>_test
```

- Build and Run performance test:
```
  $ ninja libc.test.src.math.differential_testing.<func>_perf
  $ projects/libc/test/src/math/differential_testing/libc.test.src.math.differential_testing.<func>_perf
  $ cat <func>_perf.log
```

## Code reviews

We follow the code review process of LLVM with Phabricator:
```
  https://llvm.org/docs/Phabricator.html
```
