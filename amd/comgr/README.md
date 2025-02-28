Code Object Manager (Comgr)
===========================

The Comgr library provides APIs for compiling and inspecting AMDGPU code
objects. The API is documented in the [header file](include/amd_comgr.h.in).

Building the Code Object Manager
--------------------------------

Comgr depends on [LLVM](https://github.com/ROCm/llvm-project) and
[AMDDeviceLibs](https://github.com/ROCm/llvm-project/tree/amd-staging/amd/device-libs).
One way to make these visible to the Comgr build process is by setting the
`CMAKE_PREFIX_PATH` to include either the build directory or install prefix of
each of these components, separated by a semicolon. Both should be built using
either sources with the same ROCm release tag, or from the `amd-staging`
branch. LLVM should be built with at least
`LLVM_ENABLE_PROJECTS='llvm;clang;lld'` and
`LLVM_TARGETS_TO_BUILD='AMDGPU;X86'`.

An example `bash` session to build Comgr on Linux using GNUMakefiles is:

    $ LLVM_PROJECT=~/llvm-project
    $ DEVICE_LIBS=~/llvm-project/amd/device-libs
    $ COMGR=~/llvm-project/amd/comgr
    $ mkdir -p "$LLVM_PROJECT/build"
    $ cd "$LLVM_PROJECT/build"
    $ cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_PROJECTS="llvm;clang;lld" \
        -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" \
        ../llvm
    $ make
    $ mkdir -p "$DEVICE_LIBS/build"
    $ cd "$DEVICE_LIBS/build"
    $ cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$LLVM_PROJECT/build" \
        ..
    $ make
    $ mkdir -p "$COMGR/build"
    $ cd "$COMGR/build"
    $ cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$LLVM_PROJECT/build;$DEVICE_LIBS/build" \
        ..
    $ make
    $ make test

The equivalent on Windows in `cmd.exe` using Visual Studio project files is:

    > set LLVM_PROJECT="%HOMEPATH%\llvm-project"
    > set DEVICE_LIBS="%HOMEPATH%\llvm-project\amd\device-libs"
    > set COMGR="%HOMEPATH%\llvm-project\amd\comgr"
    > mkdir "%LLVM_PROJECT%\build"
    > cd "%LLVM_PROJECT%\build"
    > cmake ^
        -DLLVM_ENABLE_PROJECTS="llvm;clang;lld" ^
        -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" ^
        ..\llvm
    > msbuild /p:Configuration=Release ALL_BUILD.vcxproj
    > mkdir "%DEVICE_LIBS%\build"
    > cd "%DEVICE_LIBS%\build"
    > cmake ^
        -DCMAKE_PREFIX_PATH="%LLVM_PROJECT%\build" ^
        ..
    > msbuild /p:Configuration=Release ALL_BUILD.vcxproj
    > mkdir "%COMGR%\build"
    > cd "%COMGR%\build"
    > cmake ^
        -DCMAKE_PREFIX_PATH="%LLVM_PROJECT%\build;%DEVICE_LIBS%\build" ^
        ..
    > msbuild /p:Configuration=Release ALL_BUILD.vcxproj
    > msbuild /p:Configuration=Release RUN_TESTS.vcxproj

Optionally,
[AddressSanitizer](https://github.com/google/sanitizers/wiki/AddressSanitizer)
may be enabled during development via `-DADDRESS_SANITIZER=On` during the Comgr
`cmake` step.

Comgr can be built as a static library by passing
`-DCOMGR_BUILD_SHARED_LIBS=OFF` during the Comgr `cmake` step.

Comgr SPIRV-related APIs can be disabled by passing
`-DCOMGR_DISABLE_SPIRV=1` during the Comgr `cmake` step. This removes any
dependency on LLVM SPIRV libraries or the llvm-spirv tool.

Depending on the Code Object Manager
------------------------------------

Comgr exports a CMake package named `amd_comgr` for both the build and install
trees. This package defines a library target named `amd_comgr`. To depend on
this in your CMake project, use `find_package`:

    find_package(amd_comgr REQUIRED CONFIG)
    ...
    target_link_libraries(your_target amd_comgr)

If Comgr is not installed to a standard CMake search directory, the path to the
build or install tree can be supplied to CMake via `CMAKE_PREFIX_PATH`:

    cmake -DCMAKE_PREFIX_PATH=path/to/comgr/build/or/install

Environment Variables
---------------------

Comgr lazily evaluates certain environment variables when their value is first
required. If the value is used, it is read once at the time it is needed, and
then cached. The exact behavior when changing these values during the execution
of a process after Comgr APIs have been invoked is undefined.

Comgr supports some environment variables to locate parts of the ROCm stack.
These include:

* `ROCM_PATH`: If this is set it is used as an absolute path to the root of the
  ROCm installation, which is used when determining the default values for
  `HIP_PATH` and `LLVM_PATH` (see below). If this is not set and if a ROCM
  package is provided during the build, ROCM path is set to it. Otherwise Comgr
  tries to construct the ROCM path from the path where amd_comgr library
  is located.
* `HIP_PATH`: If this is set it is used as an absolute path to the root of the
  HIP installation. If this is not set, it has a default value of
  "${ROCM_PATH}/hip".
* `LLVM_PATH`: If this is set it is used as an absolute path to the root of the
  LLVM installation, which is currently used for HIP compilation to locate
  certain runtime headers. If this is not set, it has a default value of
  "${ROCM_PATH}/llvm".

Comgr utilizes a cache to preserve the results of compilations between executions.
The cache's status (enabled/disabled), storage location for its results,
and eviction policy can be manipulated through specific environment variables.
If an issue arises during cache initialization, the execution will proceed with
the cache turned off.

By default, the cache is turned off, set the environment variable
`AMD_COMGR_CACHE=1` to enable it. This may change in a future release.

* `AMD_COMGR_CACHE`: When unset or set to 0, the cache is turned off.
* `AMD_COMGR_CACHE_DIR`: When set to "", the cache is turned off. If assigned a
  value, that value is used as the path for cache storage. By default, it is
  directed to "$XDG_CACHE_HOME/comgr_cache" (which defaults to
  "$USER/.cache/comgr_cache" on Linux, and "%LOCALAPPDATA%\cache\comgr_cache"
  on Microsoft Windows).
* `AMD_COMGR_CACHE_POLICY`: If assigned a value, the string is interpreted and
  applied to the cache pruning policy. The cache is pruned only upon program
  termination. The string format aligns with [Clang's ThinLTO cache pruning policy](https://clang.llvm.org/docs/ThinLTO.html#cache-pruning).
  The default policy is set as: "prune_interval=1h:prune_expiration=0h:cache_size=75%:cache_size_bytes=30g:cache_size_files=0".

Comgr also supports some environment variables to aid in debugging. These
include:

* `AMD_COMGR_SAVE_TEMPS`: If this is set, and is not "0", Comgr does not delete
  temporary files generated during compilation. These files do not appear in
  the current working directory, but are instead left in a platform-specific
  temporary directory (typically `/tmp` on Linux and `C:\Temp` or the path
  found in the `TEMP` environment variable on Windows).
* `AMD_COMGR_REDIRECT_LOGS`: If this is not set, or is set to "0", logs are
  returned to the caller as normal. If this is set to "stdout"/"-" or "stderr",
  logs are instead redirected to the standard output or error stream,
  respectively. If this is set to any other value, it is interpreted as a
  filename which logs should be appended to.
* `AMD_COMGR_EMIT_VERBOSE_LOGS`: If this is set, and is not "0", logs will
  include additional Comgr-specific informational messages.
* `AMD_COMGR_TIME_STATISTICS`: If this is set, and is not "0", logs will
  include additional Comgr-specific timing information for compilation actions.

Versioning
----------

Comgr is versioned according to a `major.minor` number scheme. The version of
the library can be determined dynamically via the `amd_comgr_get_version`
function. The version is not changed due to bug-fixes. The minor version number
is incremented for each backwards-compatible change introduced. The major
version number is incremented, and the minor version is reset to zero, for each
backwards-incompatible change introduced. Information about Comgr changes
can be found in the [release notes](docs/ReleaseNotes.md).

ISA Metadata and Versioning
---------------------------

Comgr supports multiple instruction set architectures (ISA) and APIs to query
metadata associated with an ISA. The queried metadata follows a semantic
versioning scheme e.g. major.minor.patch. The major version changes signifies
backward incompatible changes.

* `1.0.0` : Support for new target feature syntax introduced at [AMDGPUUsage](https://llvm.org/docs/AMDGPUUsage.html).
  Metadata query for a bare ISA string now returns the supported target
  features along with other details. A new key for the version is introduced.
* `0.0.x` : Support for querying the metadata for an ISA. The metadata is
  supplied in a map format with details of target triple, features and
  resource limits associated with registers and memory addressing. The
  version key is absent in the Metadata.

Thread Saftey
-------------

Comgr strives to be thread-safe when called from multiple threads in the same
process. Because of complications from a shared global state in LLVM, to
accomplish this Comgr internally implements locking mechanisms around LLVM-based
actions.

Although the locks in Comgr can allow independent actions to be safely executed
in a multithreaded environment, the user-code must still guard against
concurrent method calls which may access any particular Comgr object's state.
A Comgr object shared between threads is only safe to use as long as each thread
carefully locks out access by any other thread while it uses the shared object.

Coding Standards
----------------

Wherever possible, Comgr adheres to the same coding standards as
[LLVM](https://llvm.org/docs/CodingStandards.html). Comgr also includes
configuration files for
[clang-format](https://clang.llvm.org/docs/ClangFormat.html) and
[clang-tidy](https://clang.llvm.org/extra/clang-tidy/), which should be used to
ensure patches conform.

A script at `utils/tidy-and-format.sh` can be run to help automate the task of
ensuring all sources conform to the coding standards. To support the use of
this script, any exceptions must be annotated in source comments, as described
in the clang-tidy manual.
