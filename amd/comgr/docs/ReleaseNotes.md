Comgr v3.0 Release Notes
========================

This document contains the release notes for the Code Object Manager (Comgr),
part of the ROCm Software Stack, release v3.0. Here we describe the status of
Comgr, including major improvements from the previous release and new feature

These are in-progress notes for the upcoming Comgr v3.0 release.
Release notes for previous releases can be found in
[docs/historical](docs/historical).

Potentially Breaking Changes
----------------------------
These changes are ones which we think may surprise users when upgrading to
Comgr v3.0 because of the opportunity they pose for disruption to existing
code bases.

-  Removed -h option from comgr-objdump: The -h option (short for -headers) is a
legal comgr-objdump option. However registering this as an LLVM option by Comgr
prevents other LLVM tools or instances from registering a -h option in the same
process, which is an issue because -h is a common short form for -help.
-  Updated default code object version used when linking code object specific
device library from v4 to v5

New Features
------------
- Added support for linking code\_object\_v4/5 device library files.
- Enabled llvm dylib builds. When llvm dylibs are enabled, a new package
rocm-llvm-core will contain the required dylibs for Comgr.
- Moved build to C++17, allowing us to use more modern features in the
implementation and tests.
- Enabled thread-safe execution of Comgr by enclosing primary Comgr actions in
an std::scoped\_lock()
- Added support for bitcode and archive unbundling during linking via the new
llvm OffloadBundler API.

Bug Fixes
---------
- Fixed symbolizer assertion for non-null terminated file-slice content,
by bypassing null-termination check in llvm::MemoryBuffer
- Fixed bug and add error checking for internal unbundling. Previously internal
unbundler would fail if files weren't already present in filesystem.
- Fixed issue where lookUpCodeObject() would fail if code object ISA strings
weren't listed in order.
- Added support for subdirectories in amd\_comgr\_set\_data\_name(). Previously
names with a "/" would generate a file-not-found error.
- Added amdgpu-internalize-symbols option to bitcode codegen action, which has
significant performance implications
- Fixed an issue where -nogpulib was always included in HIP compilations, which
prevented correct execution of
COMPILE\_SOURCE\_WITH\_DEVICE\_LIBS\_TO\_BC action.
- Fixed a multi-threading bug where programs would hang when calling Comgr APIs
like amd\_comgr\_iterate\_symbols() from multiple threads
- Fixed an issue where providing DataObjects with an empty name to the bitcode
linking action caused errors when AMD\_COMGR\_SAVE\_TEMPS was enabled, or when
linking bitcode bundles.
- Updated to use lld::lldMain() introduced in D110949 instead of the older
lld::elf::link in Comgr's linkWithLLD()
- Added -x assembler option to assembly compilation. Before, if an assembly file
did not end with a .s file extension, it was not handled properly by the Comgr
ASSEMBLE\_SOURCE\_TO\_RELOCATABLE action.
- Switched getline() from C++ to C-style to avoid issues with stdlibc++ and
pytorch


New APIs
--------
- amd\_comgr\_populate\_mangled\_names() (v2.5)
- amd\_comgr\_get\_mangled\_name() (v2.5)
    - Support bitcode and executable name lowering. The first call populates a
    list of mangled names for a given data object, while the second fetches a
    name from a given object and index.
- amd\_comgr\_populate\_name\_expression\_map() (v2.6)
- amd\_comgr\_map\_name\_expression\_to\_symbol\_name() (v2.6)
    - Support bitcode and code object name expression mapping. The first call
    populates a map of name expressions for a given comgr data object, using
    LLVM APIs to traverse the bitcode or code object. The second call returns
    a value (mangled symbol name) from the map for a given key (unmangled
    name expression). These calls assume that names of interest have been
    enclosed the HIP runtime using a stub attribute containg the following
    string in the name: "__amdgcn_name_expr".
- amd\_comgr\_map\_elf\_virtual\_address\_to\_code\_object\_offset() (v2.7)
    - For a given executable and ELF virtual address, return a code object
    offset. This API will benifet the ROCm debugger and profilier


Deprecated APIs
---------------

Removed APIs
------------

New Comgr Actions and Data Types
--------------------------------
- (Action) AMD\_COMGR\_ACTION\_COMPILE\_SOURCE\_TO\_RELOCATABLE
  - This action performs compile-to-bitcode, linking device libraries, and
codegen-to-relocatable in a single step. By doing so, clients are able to defer more
of the flag handling to toolchain. Currently only supports HIP.
- (Data Type) AMD\_COMGR\_DATA\_KIND\_BC\_BUNDLE
- (Data Type) AMD\_COMGR\_DATA\_KIND\_AR\_BUNDLE
  - These data kinds can now be passed to an AMD\_COMGR\_ACTION\_LINK\_BC\_TO\_BC
action, and Comgr will internally unbundle and link via the OffloadBundler and linkInModule APIs.
- (Language Type) AMD\_COMGR\_LANGUAGE\_LLVM\_IR
  - This language can now be passed to AMD\_COMGR\_ACTION\_COMPILE\_\* actions
  to enable compilation of LLVM IR (.ll or .bc) files. This is useful for MLIR
  contexts.
- (Action) AMD\_COMGR\_ACTION\_COMPILE\_SOURCE\_TO\_EXECUTABLE
  - This action allows compilation from source directly to executable, including
  linking device libraries.


Deprecated Comgr Actions and Data Types
---------------------------------------

Removed Comgr Actions and Data Types
------------------------------------

Comgr Testing, Debugging, and Logging Updates
---------------------------------------------
- Added support for C++ tests. Although Comgr APIs are C-compatible, we can now
use C++ features in testing (C++ threading APIs, etc.)
- Clean up test directory by moving sources to subdirectory
- Several tests updated to pass while verbose logs are redirected to stdout
- Log information reported when AMD\_COMGR\_EMIT\_VERBOSE\_LOGS updated to:
    - Show both user-facing clang options used (Compilation Args) and internal
    driver options (Driver Job Args)
    - Show files linked by linkBitcodeToBitcode()
- Remove support for code object v2 compilation in tests and test CMAKE due to
deprecation of code object v2 in LLVM. However, we still test loading and
metadata querys for code object v2 objects.
- Remove support for code object v3 compilation in tests and test CMAKE due to
deprecation of code object v3 in LLVM. However, we still test loading and
metadata querys for code object v3 objects.
- Revamp symbolizer test to fail on errors, among other improvments
- Improve linking and unbundling log to correctly store temporary files in /tmp,
and to output clang-offload-bundler command to allow users to re-create Comgr
unbundling.
- Add git branch and commit hash for Comgr, and commit hash for LLVM to log
output for Comgr actions. This can help us debug issues more quickly in cases
where reporters provide Comgr logs.
- Fix multiple bugs with mangled names test
- Update default arch for test binaries from gfx830 to gfx900
- Refactor nested kernel behavior into new test, as this behavior is less common
and shouldn't be featured in the baseline tests

New Targets
-----------
 - gfx940
 - gfx941
 - gfx942
 - gfx1036
 - gfx1150
 - gfx1151

Removed Targets
---------------

Significant Known Problems
--------------------------
- Several Comgr actions currently write and read files from the filesystem,
which is a known performance issue. We aim to address this by improving
clang's virtual file system support
- Several Comgr actions currently fork new processes for compilation actions. We
aim to address this by librayizing llvm tools that are currently only useable as
a separate process.
