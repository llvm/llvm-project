====================
Clang Linker Wrapper
====================

.. contents::
   :local:

.. _clang-linker-wrapper:

Introduction
============

This tool works as a wrapper of the normal host linking job. This tool is used
to create linked device images for offloading and the necessary runtime calls to
register them. It works by first scanning the linker's input for embedded device
offloading data stored at the ``.llvm.offloading`` section. This section
contains binary data created by the :doc:`ClangOffloadPackager`. The extracted
device files will then be linked. The linked modules will then be wrapped into a
new object file containing the code necessary to register it with the offloading
runtime.

Usage
=====

This tool can be used with the following options. Any arguments not intended
only for the linker wrapper will be forwarded to the wrapped linker job.

.. code-block:: console

  USAGE: clang-linker-wrapper [options] -- <options to passed to the linker>

  OPTIONS:
    --bitcode-library=<kind>-<triple>-<arch>=<path>
                           Extra bitcode library to link
    --cuda-path=<dir>      Set the system CUDA path
    --device-debug         Use debugging
    --device-linker=<value> or <triple>=<value>
                           Arguments to pass to the device linker invocation
    --dry-run              Print program arguments without running
    --embed-bitcode        Embed linked bitcode in the module
    --help-hidden          Display all available options
    --help                 Display available options (--help-hidden for more)
    --host-triple=<triple> Triple to use for the host compilation
    --linker-path=<path>   The linker executable to invoke
    -L <dir>               Add <dir> to the library search path
    -l <libname>           Search for library <libname>
    --opt-level=<O0, O1, O2, or O3>
                           Optimization level for LTO
    -o <path>              Path to file to write output
    --pass-remarks-analysis=<value>
                           Pass remarks for LTO
    --pass-remarks-missed=<value>
                           Pass remarks for LTO
    --pass-remarks=<value> Pass remarks for LTO
    --print-wrapped-module Print the wrapped module's IR for testing
    --ptxas-arg=<value>    Argument to pass to the 'ptxas' invocation
    --save-temps           Save intermediate results
    --sysroot<value>       Set the system root
    --verbose              Verbose output from tools
    --v                    Display the version number and exit
    --                     The separator for the wrapped linker arguments


Example
=======

This tool links object files with offloading images embedded within it using the
``-fembed-offload-object`` flag in Clang. Given an input file containing the
magic section we can pass it to this tool to extract the data contained at that
section and run a device linking job on it.

.. code-block:: console

  clang-linker-wrapper --host-triple=x86_64 --linker-path=/usr/bin/ld -- <Args>
