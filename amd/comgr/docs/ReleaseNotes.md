Comgr v4.0 Release Notes
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

Bug Fixes
---------

New APIs
--------

Deprecated APIs
---------------

Removed APIs
------------

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
