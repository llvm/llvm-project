Code Object Manager (Comgr)
===========================

The Comgr library provides APIs for compiling and inspecting AMDGPU code
objects. The API is documented in the [header file](include/amd_comgr.h.in).
The Comgr API is compatible with C99 and C++.

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

    $ LLVM_PROJECT=~/llvm-project/build
    $ DEVICE_LIBS=~/llvm-project/amd/device-libs/build
    $ mkdir -p "$LLVM_PROJECT"
    $ cd "$LLVM_PROJECT"
    $ cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_PROJECTS="llvm;clang;lld" \
        -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" \
        ../llvm
    $ make
    $ mkdir -p "$DEVICE_LIBS"
    $ cd "$DEVICE_LIBS"
    $ cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$LLVM_PROJECT" \
        ..
    $ make
    $ cd ~/llvm-project/amd/comgr
    $ mkdir -p build; cd build;
    $ cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$LLVM_PROJECT;$DEVICE_LIBS" \
        ..
    $ make
    $ make test

The equivalent on Windows in `cmd.exe` using Visual Studio project files is:

    > set LLVM_PROJECT="%HOMEPATH%\llvm-project\build"
    > set DEVICE_LIBS="%HOMEPATH%\llvm-project\amd\device-libs\build"
    > mkdir "%LLVM_PROJECT%"
    > cd "%LLVM_PROJECT%"
    > cmake ^
        -DLLVM_ENABLE_PROJECTS="llvm;clang;lld" ^
        -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" ^
        ..\llvm
    > msbuild /p:Configuration=Release ALL_BUILD.vcxproj
    > mkdir "%DEVICE_LIBS%"
    > cd "%DEVICE_LIBS%"
    > cmake ^
        -DCMAKE_PREFIX_PATH="%LLVM_PROJECT%" ^
        ..
    > msbuild /p:Configuration=Release ALL_BUILD.vcxproj
    > cd "%HOMEPATH%\llvm-project\amd\comgr"
    > mkdir build
    > cd build
    > cmake ^
        -DCMAKE_PREFIX_PATH="%LLVM_PROJECT%;%DEVICE_LIBS%" ^
        ..
    > msbuild /p:Configuration=Release ALL_BUILD.vcxproj
    > msbuild /p:Configuration=Release RUN_TESTS.vcxproj

**ASAN support:** Optionally,
[AddressSanitizer](https://github.com/google/sanitizers/wiki/AddressSanitizer)
may be enabled during development via `-DADDRESS_SANITIZER=On` during the Comgr
`cmake` step.

**Static Comgr:** Comgr can be built as a static library by passing
`-DCOMGR_BUILD_SHARED_LIBS=OFF` during the Comgr `cmake` step.

**SPIRV Support:** To enable SPIRV support, checkout
[SPIRV-LLVM-Translator](https://github.com/ROCm/SPIRV-LLVM-Translator) in
`llvm/projects` or `llvm/tools` and build using the above instructions, with the
exception that the `-DCMAKE_PREFIX_PATH` for llvm-project must be an install
path (specified with `-DCMAKE_INSTALL_PREFIX=/path/to/install/dir` and populated
with `make install`) rather than the build path.

Comgr SPIRV-related APIs can be disabled by passing
`-DCOMGR_DISABLE_SPIRV=1` during the Comgr `cmake` step. This removes any
dependency on LLVM SPIRV libraries or the llvm-spirv tool.

**Code Coverage Instrumentation:** Comgr supports source-based [code coverage
via clang](https://clang.llvm.org/docs/SourceBasedCodeCoverage.html), and
leverages the same CMake variables as
[LLVM](https://www.llvm.org/docs/CMake.html#llvm-related-variables)
(LLVM_BUILD_INSTRUMENTED_COVERAGE, etc.).

Example of insturmenting with covereage, generating profiles, and creating an
HTML for investigation:

    $ cmake -DCMAKE_STRIP="" -DLLVM_PROFILE_DATA_DIR=`pwd`/profiles \
        -DLLVM_BUILD_INSTRUMENTED_COVERAGE=On \
        -DCMAKE_CXX_COMPILER="$LLVM_PROJECT/bin/clang++" \
        -DCMAKE_C_COMPILER="$LLVM_PROJECT/bin/clang" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$LLVM_PROJECT;$DEVICE_LIBS" ..
    $ make -j
    $ make test test-lit
    $ cd profiles
    # Manually aggregate the data and create text report.
    $ $LLVM_PROJECT/bin/llvm-profdata merge -sparse *.profraw -o \
        comgr_test.profdata # merge and index data
    $ $LLVM_PROJECT/bin/llvm-cov report ../libamd_comgr.so \
        -instr-profile=comgr_test.profdata \
        -ignore-filename-regex="[cl].*/include/*" # show test report without \
        includes
    # Or use python script to aggregate the data and create html report.
    $ $LLVM_PROJECT/../llvm/utils/prepare-code-coverage-artifact.py \
        --preserve-profiles $LLVM_PROJECT/bin/llvm-profdata \
        $LLVM_PROJECT/bin/llvm-cov . html ../libamd_comgr.so \
        # create html report

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

Testing
--------------------------------

Comgr has both unit tests (older) and LLVM LIT tests (newer). They can be run
from the build directory via:

    make test # unit
    make test-lit # lit

Environment Variables
---------------------

Comgr lazily evaluates certain environment variables when their value is first
required. If the value is used, it is read once at the time it is needed, and
then cached. The exact behavior when changing these values during the execution
of a process after Comgr APIs have been invoked is undefined.

Comgr supports an environment variable to help locate LLVM:

* `LLVM_PATH`: If set, it is used as an absolute path to the root of the LLVM
  installation, which is currently used to locate the clang resource directory
  and clang binary path, allowing for additional optimizations.

### Caching
Comgr utilizes a cache to preserve the results of compilations between executions.
The cache's status (enabled/disabled), storage location for its results,
and eviction policy can be manipulated through specific environment variables.
If an issue arises during cache initialization, the execution will proceed with
the cache turned off.

By default, the cache is enabled.

* `AMD_COMGR_CACHE`: When unset or set to a value different than "0", the cache is enabled.
  Disabled when set to "0".
* `AMD_COMGR_CACHE_DIR`: If assigned a non-empty value, that value is used as
  the path for cache storage. If the variable is unset or set to an empty string `""`,
  it is directed to "$XDG_CACHE_HOME/comgr" (which defaults to
  "$USER/.cache/comgr" on Linux, and "%LOCALAPPDATA%\cache\comgr"
  on Microsoft Windows).
* `AMD_COMGR_CACHE_POLICY`: If assigned a value, the string is interpreted and
  applied to the cache pruning policy. The cache is pruned only upon program
  termination. The string format aligns with [Clang's ThinLTO cache pruning policy](https://clang.llvm.org/docs/ThinLTO.html#cache-pruning).
  The default policy is set as: "prune_interval=1h:prune_expiration=0h:cache_size=75%:cache_size_bytes=30g:cache_size_files=0".

### Debugging
Comgr supports some environment variables to aid in debugging. These
include:

* `AMD_COMGR_SAVE_TEMPS`: If this is set, and is not "0", Comgr does not delete
  temporary files generated during compilation. These files do not appear in
  the current working directory, but are instead left in a platform-specific
  temporary directory (typically `/tmp` on Linux and `C:\Temp` or the path
  found in the `TEMP` environment variable on Windows).
* `AMD_COMGR_SAVE_LLVM_TEMPS`: If this is set, Comgr forwards `--save-temps=obj`
  to Clang Driver invocations.
* `AMD_COMGR_REDIRECT_LOGS`: If this is not set, or is set to "0", logs are
  returned to the caller as normal. If this is set to "stdout"/"-" or "stderr",
  logs are instead redirected to the standard output or error stream,
  respectively. If this is set to any other value, it is interpreted as a
  filename which logs should be appended to.
* `AMD_COMGR_EMIT_VERBOSE_LOGS`: If this is set, and is not "0", logs will
  include additional Comgr-specific informational messages.
* `AMD_COMGR_TIME_STATISTICS`: If this is set, and is not "0", logs will
  include additional Comgr-specific timing information for compilation actions.

### VFS
Comgr implements support for an in-memory, virtual filesystem (VFS) for storing
temporaries generated during intermediate compilation steps. This is aimed at 
improving performance by reducing on-disk file I/O. Currently, VFS is only supported 
for the device library link step, but we aim to progressively add support for
more actions.

By default, VFS is turned on.

* `AMD_COMGR_USE_VFS`: When set to "0", VFS support is turned off.
* Users may use the API `amd_comgr_action_info_set_vfs` to disable VFS for individual actions
  without having to modify system-wide environment variables.
* If `AMD_COMGR_SAVE_TEMPS` is set and not "0", VFS support is turned off irrespective
  of `AMD_COMGR_USE_VFS` or the use of `amd_comgr_action_info_set_vfs`.

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

Aligning with the purpose of being a stable interface into LLVM functionality,
the core enum values (AMD\_COMGR\_LANGUAGE_\*, AMD\_COMGR\_DATA\_KIND\_\*,
AMD\_COMGR\_ACTION\_\*, etc.) should remain consistent between versions, even if
some enum values are deprecated and removed. This will avoid potential breakages
and binary incompatibilities.
