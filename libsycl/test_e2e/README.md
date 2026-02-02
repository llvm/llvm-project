## Getting Started

This directory contains `libsycl` tests distributed in subdirectories based on
testing scope. `libsycl` uses LIT to configure and run its tests.

Please see the [Lit Command Guide](https://llvm.org/docs/CommandGuide/lit.html)
for more information about LIT.

## Prerequisites

* Target runtime(s) to execute tests on devices.
  TODO: add link to liboffload instruction once they add it.
* Compiler & libsycl. Can be built following these
  [instructions](/libsycl/docs/index.rst).

## Run the tests

`libsycl` is integrated via LLVM_ENABLE_RUNTIMES and is not visible as top
level target. Same is applicable for tests. To run `check-sycl-e2e` tests you
need to prefix `<build>/runtimes/runtimes-bins/` to the paths of all tests.
For example, to run all the libsycl end-to-end tests you can do:
```bash
<build>/bin/llvm-lit <build>/runtimes/runtimes-bins/libsycl/test_e2e
```

To run individual test, use the path to it instead.

If you are using `ninja` as your build system, you can run all the tests in the
libsycl testsuite as:

```bash
 ninja -C <build>/runtimes/runtimes-bins check-sycl-e2e
 ```


## CMake parameters

These parameters can be used to configure tests:

`LIBSYCL_CXX_COMPILER` - path to compiler to use it for building tests.

`LIBSYCL_E2E_CXX_FLAGS` - flags to be passed to `LIBSYCL_CXX_COMPILER` when
    building libsycl end-to-end tests.

`LLVM_LIT` - path to llvm-lit tool.

## Creating or modifying tests

### LIT feature checks

Following features can be passed to LIT via `REQUIRES`, `UNSUPPORTED`, etc.
filters to limit test execution to the specific environment.

#### Auto-detected features

The following features are automatically detected by `llvm-lit` by scanning the
environment:

* `linux` - host OS;
* `any-device-is-gpu` - device type to be available;
* `any-device-is-level_zero` - backend to be available;

Note: `sycl-ls` tool doesn't have assigned feature since it is essential for
tests configuration and is always available if test is executed.

### llvm-lit parameters

Following options can be passed to `llvm-lit` tool with `--param` option to
configure test execution:

* `libsycl_compiler` - full path to compiler to use;
* `extra_environment` - comma-separated list of variables with values to be
  added to test environment. Can be also set by `LIT_EXTRA_ENVIRONMENT`
  variable in CMake.
* `extra_system_environment` - comma-separated list of variables to be
  propagated from the host environment to test environment. Can be also set by
  `LIT_EXTRA_SYSTEM_ENVIRONMENT` variable in CMake.

Example:

```bash
<build>/bin/llvm-lit  --param libsycl_compiler=path/to/clang++ \
        <build>/runtimes/runtimes-bins/libsycl/test_e2e
```