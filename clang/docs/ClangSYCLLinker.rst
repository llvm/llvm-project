=======================
Clang SYCL Linker
=======================

.. contents::
   :local:

.. _clang-sycl-linker:

Introduction
============

This tool works as a wrapper around the SYCL device code linking process.
The purpose of this tool is to provide an interface to link SYCL device bitcode
in LLVM IR format together with LLVM bitcode libraries, and then run SPIR-V
code generation on the fully linked device code to produce the final output.
After the linking stage, the fully linked device code in LLVM IR format may
undergo several SYCL-specific finalization steps before the SPIR-V code
generation step.
The tool will also support the Ahead-Of-Time (AOT) compilation flow. AOT
compilation is the process of invoking the back-end at compile time to produce
the final binary, as opposed to just-in-time (JIT) compilation when final code
generation is deferred until application runtime.

Device code linking for SYCL offloading has known quirks that make it
difficult to use in a unified offloading setting. The primary issue is that
several finalization steps are required to be run on the fully linked LLVM
IR bitcode to guarantee conformance to SYCL standards. This step is unique to
the SYCL offloading compilation flow.

This tool has been proposed to work around this issue.

Usage
=====

This tool can be used with the following options. Several of these options will
be passed down to downstream AOT compilation tools like 'ocloc' and 'opencl-aot'.

.. code-block:: console

  OVERVIEW: A utility that wraps around the SYCL device code linking process.
  This enables LLVM IR linking, post-linking and code generation for SPIR-V
  JIT and AOT targets.

  USAGE: clang-sycl-linker [options] <input bitcode files>

  OPTIONS:
    --arch <value>                Specify the name of the target architecture.
    --dry-run                     Print generated commands without running.
    -help-hidden                  Display all available options
    -help                         Display available options (--help-hidden for more)
    -L <dir>                      Add <dir> to the library search path
    --bc-library <name>           Search for LLVM bitcode library <name> (provided with extension, e.g. --bc-library foo.bc)
    --module-split-mode=<mode>    Module split mode: 'source' (default), 'kernel', or 'none'
    --ocloc-options=<value>       Options passed to ocloc for Intel GPU AOT compilation
    --opencl-aot-options=<value>  Options passed to opencl-aot for Intel CPU AOT compilation
    -o <path>                     Path to file to write output
    --save-temps                  Save intermediate results
    --triple <value>              Specify the target triple.
    --version                     Display the version number and exit
    -v                            Print verbose information
    -spirv-dump-device-code=<dir> Directory to dump SPIR-V IR code into

Example
=======

This tool is intended to be invoked when targeting any of the target offloading
toolchains. When the --sycl-link option is passed to the clang driver, the
driver will invoke the linking job of the target offloading toolchain, which in
turn will invoke this tool. This tool can be used to create one or more fully
linked device images that are ready to be wrapped and linked with host code to
generate the final executable.

.. code-block:: console

  clang-sycl-linker --triple spirv64 --arch bmg_g21 input.bc
