(build_and_test)=

# Building and Testing the libc

## Build modes

The libc can be built and tested in two different modes:

1. **The overlay mode** - In this mode, one uses the static archive from LLVM's
   libc along with the system libc. See {ref}`overlay_mode` for more details
   on building and using the libc in this mode. You can only run the libc
   unittests in this mode. To run them, one simply does:

   ```sh
   $> ninja check-libc
   ```

   Note that, unittests for only those functions which are part of the overlay
   static archive will be run with the above command.

2. **The full build mode** - In this mode, the libc is used as the only libc
   for the user's application. See {ref}`full_host_build` for more details on
   building and using the libc in this mode. Once configured for a full libc
   build, you can run three kinds of tests:

   1. Unit tests - You can run unittests by the command:

      ```sh
      $> ninja check-libc
      ```

   2. Integration tests - You can run integration tests by the command:

      ```sh
      $> ninja libc-integration-tests
      ```

   3. Shared tests - You can run tests for shared, standalone components (like math primitives) without needing the full libc runtime by the command:

      ```sh
      $> ninja libc-shared-tests
      ```

## Building with VSCode

As a quickstart to using VSCode for development, install the cmake extension
and put the following in your settings.json file:

```javascript
{
  "cmake.sourceDirectory": "${workspaceFolder}/runtimes",
  "cmake.configureSettings": {
    "LLVM_ENABLE_RUNTIMES" : ["libc", "compiler-rt"],
    "LLVM_LIBC_FULL_BUILD" : true,
    "LLVM_ENABLE_SPHINX" : true,
    "LIBC_INCLUDE_DOCS" : true,
    "LLVM_LIBC_INCLUDE_SCUDO" : true,
    "COMPILER_RT_BUILD_SCUDO_STANDALONE_WITH_LLVM_LIBC": true,
    "COMPILER_RT_BUILD_GWP_ASAN" : false,
    "COMPILER_RT_SCUDO_STANDALONE_BUILD_SHARED" : false,
    "CMAKE_EXPORT_COMPILE_COMMANDS" : true,
    "LIBC_CMAKE_VERBOSE_LOGGING" : true
  }
}
```

## Building with Bazel

1. To build with Bazel, use the following command:

   ```sh
   $> bazel build --config=generic_clang @llvm-project//libc/...
   ```

1. To run the unit tests with bazel, use the following command:

   ```sh
   $> bazel test --config=generic_clang @llvm-project//libc/...
   ```

1. The bazel target layout of `libc` is located at: [utils/bazel/llvm-project-overlay/libc/BUILD.bazel](https://github.com/llvm/llvm-project/tree/main/utils/bazel/llvm-project-overlay/libc/BUILD.bazel).

## Building in a container for a different architecture

[Podman](https://podman.io/) can be used together with
[QEMU](https://www.qemu.org/) to run container images built for architectures
other than the host's. This can be used to build and test the libc on other
supported architectures for which you do not have access to hardware. It can
also be used if the hardware is slower than emulation of its architecture on a
more powerful machine under a different architecture.

As an example, to build and test in a container for 32-bit Arm:

1. To install the necessary packages on Arch Linux:

   ```sh
   $> pacman -S podman qemu-user-static qemu-user-static-binfmt \
      qemu-system-arm
   ```

2. To run Bash interactively in an Ubuntu 22.04 container for 32-bit Arm and
   bind-mount an existing checkout of llvm-project on the host:

   ```sh
   $> podman run -it \
      -v </host/path/to/llvm-project>:</container/path/to/llvm-project> \
      --arch arm docker.io/ubuntu:jammy bash
   ```

3. Install necessary packages, invoke CMake, build, and run tests.

## Building and Testing with an Emulator

If you are cross-compiling the libc for a different architecture, you can use an emulator
such as QEMU to run the tests directly on your host without a container. See
{ref}`full_cross_build` for detailed instructions on configuring CMake to use an emulator.
