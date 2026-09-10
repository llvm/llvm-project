<!-- If you want to modify sections/contents permanently, you should modify both
ReleaseNotes.md and ReleaseNotesTemplate.txt. -->

(lld-release-release-notes)=

# lld {{ release | default("") }} Release Notes

```{contents}
:local: true
```

::::{only} PreRelease

:::{warning}
These are in-progress notes for the upcoming LLVM {{ release | default("") }} release.
Release notes for previous releases can be found on
[the Download Page](https://releases.llvm.org/download.html).
:::
::::

## Introduction

This document contains the release notes for the lld linker, release {{ release | default("") }}.
Here we describe the status of lld, including major improvements
from the previous release. All lld releases may be downloaded
from the [LLVM releases web site](https://llvm.org/releases/).

## Non-comprehensive list of changes in this release

### ELF Improvements

### Breaking changes

### COFF Improvements

### MinGW Improvements

### MachO Improvements

* `__objc_stubs` entries are now ordered by the priority of the sections that
  call them, so that stubs reached from prioritized code are laid out together.
  This applies whenever section priorities exist, such as with `-order_file`.

* arm64 and arm64_32 function symbols that are not guaranteed to be 4-byte
  aligned now produce a warning, matching ld64. Hand-written assembly that
  omits an explicit `.p2align 2` may need one, and links that pass
  `-fatal_warnings` may start to fail.

### WebAssembly Improvements

* Added support for resolving and merging common data symbols (allocating them
  into .bss.common in executable/shared module links, or merging them with max
  size/alignment in relocatable -r links). See
  https://github.com/WebAssembly/tool-conventions/pull/267

#### Fixes
