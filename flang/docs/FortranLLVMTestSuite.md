# Fortran Tests in the LLVM Test Suite

```{contents}
---
local:
---
```

The [LLVM Test Suite](https://github.com/llvm/llvm-test-suite) is a
separate git repo from the main LLVM project. We recommend that
first-time users read through [LLVM Test Suite
Guide](https://llvm.org/docs/TestSuiteGuide.html) which describes the
organizational structure of the test suite and how to run it.

## Running the LLVM test-suite with Fortran

Fortran support can be enabled by setting the following CMake variables:
```
cmake -G "Ninja" -DCMAKE_C_COMPILER=<path to C compiler> \
    -DCMAKE_CXX_COMPILER=<path to C++ compiler> \
    -DCMAKE_Fortran_COMPILER=<path to Fortran compiler> \
    -DTEST_SUITE_COLLECT_CODE_SIZE:STRING=OFF \
    -DTEST_SUITE_SUBDIRS:STRING="Fortran" \
    -DTEST_SUITE_FORTRAN:STRING=ON \
    -DTEST_SUITE_LIT=<path to llvm-lit> \
    <path to llvm-test-suite>
```

This will configure the test-suite to run only the Fortran tests which
are found in the Fortran subdirectory. To run the C/C++ tests
alongside the Fortran tests omit the `-DTEST_SUITE_SUBDIRS` CMake
variable.

If your Fortran compiler is Flang, there are a couple of other things you need
to do, which are explained
[here](https://github.com/llvm/llvm-test-suite/blob/main/Fortran/gfortran/README.md#usage).

Then to build and run the tests:
```
ninja
ninja check
```

## Running the SPEC CPU 2017

We recently added CMake hooks into the LLVM Test Suite to support
Fortran tests from [SPEC CPU 2017](https://www.spec.org/cpu2017/). We
strongly encourage the use of the CMake Ninja (1.10 or later) generator
due to better support for Fortran module dependency detection. Some of
the SPEC CPU 2017 Fortran tests, those that are derived from climate
codes, require support for little-endian/big-endian byte swapping
capabilities which we automatically detect at CMake configuration
time.  Note that a copy of SPEC CPU 2017 must be purchased by your
home institution and is not provided by LLVM.


Here is an example of how to build SPEC CPU 2017 with GCC

```
cmake -G "Ninja" -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_Fortran_COMPILER=gfortran \
    -DTEST_SUITE_COLLECT_CODE_SIZE:STRING=OFF \
    -DTEST_SUITE_SUBDIRS:STRING="External/SPEC" \
    -DTEST_SUITE_FORTRAN:STRING=ON \
    -DTEST_SUITE_SPEC2017_ROOT=<path to SPEC directory>  ..
```

## Running the gfortran tests

Tests from the gfortran test suite have been imported into the LLVM Test Suite.
The tests will be run automatically if the test suite is built following the
instructions described [above](#running-the-llvm-test-suite-with-fortran).
There are additional configure-time options that can be used with the gfortran 
tests. More details about those options and their purpose can be found in 
[`Fortran/gfortran/README.md`](https://github.com/llvm/llvm-test-suite/tree/main/Fortran/gfortran/README.md).
