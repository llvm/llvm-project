Comgr v4.0 (In Progress) Release Notes
========================

This document contains the release notes for the Code Object Manager (Comgr),
part of the ROCm Software Stack, release v4.0. Here we describe the status of
Comgr, including major improvements from the previous release and new feature

These are in-progress notes for the upcoming Comgr v4.0 release.
Release notes for previous releases can be found in
[docs/historical](docs/historical).

Potentially Breaking Changes
----------------------------
These changes are ones which we think may surprise users when upgrading to
Comgr v4.0 because of the opportunity they pose for disruption to existing
code bases.


New Features
------------
- Added a Comgr Caching infrastructure, currently covering the following
behaviors:
  - caching unbundling of compressed clang offload bundles
  - caching SPIR-V to LLVM IR translations
  - caching clang driver invocations
  More information about the Comgr Caching infrastructure and how to use it can
  be found in amd/comgr/README.md.
- Updated the license used for Comgr from Illinois to Apache 2.0 with LLVM
Extensions (the same license used by LLVM).
- Added Image Support to Comgr's handling of ISA metadata. Support for images
can now be queried with Comgr's metadata APIs.
- Added support for linking device library files through the use of a Virtual 
File System (VFS).

Bug Fixes
---------

New APIs
--------
- amd\_comgr\_info\_set\_vfs\_() (v3.1)
    - By setting this ActionInfo property, users can explicitly dictate if
    device libraries should be linked using the real file system or a
    Virtual File System (VFS).

Deprecated APIs
---------------

Removed APIs
------------
- The following Comgr metadata API has removed support for V2/V3 Code Objects:
  - amd\_comgr\_lookup\_code\_object()
  This API still supports Code Objects V4 and later.

New Comgr Actions and Data Types
--------------------------------

Deprecated Comgr Actions and Data Types
---------------------------------------

Removed Comgr Actions and Data Types
------------------------------------

Comgr Testing, Debugging, and Logging Updates
---------------------------------------------
- Removed HIP\_PATH and ROCM\_PATH environment variables. These were used for
now-removed Comgr actions, such as \*COMPILE\_SOURCE\_TO\_FATBIN.
- Added a new Comgr LIT testing infrastructure, which can be found in
amd/comgr/test-lit. This will allow us to write more in-depth and targeted
tests.
- Added support for source-based code coverage. See README.md for more details.

New Targets
-----------

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
