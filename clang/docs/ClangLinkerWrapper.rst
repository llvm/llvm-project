====================
Clang Linker Wrapper
====================

.. contents::
   :local:

.. _clang-linker-wrapper:

Introduction
============

This tool works as a wrapper over a linking job. The tool is used to create
linked device images for offloading. It scans the linker's input for embedded
device offloading data stored in sections ``.llvm.offloading.<triple>.<arch>``
and extracts it as a temporary file. The extracted device files will then be
passed to a device linking job to create a final device image.

Usage
=====

This tool can be used with the following options. Arguments to the host linker
being wrapper around are passed as positional arguments using the ``--`` flag to
override parsing.

.. code-block:: console

  USAGE: clang-linker-wrapper [options] <options to be passed to linker>...
  
  OPTIONS:
  
  Generic Options:
  
    --help                    - Display available options (--help-hidden for more)
    --help-list               - Display list of available options (--help-list-hidden for more)
    --version                 - Display the version of this program
  
  clang-linker-wrapper options:
  
    --host-triple=<string>    - Triple to use for the host compilation
    --linker-path=<string>    - Path of linker binary
    --opt-level=<string>      - Optimization level for LTO
    --ptxas-option=<string>   - Argument to pass to the ptxas invocation
    --save-temps              - Save intermediary results.
    --strip-sections          - Strip offloading sections from the host object file.
    --target-embed-bc         - Embed linked bitcode instead of an executable device image
    --target-feature=<string> - Target features for triple
    --target-library=<string> - Path for the target bitcode library
    -v                        - Verbose output from tools

Example
=======

This tool links object files with offloading images embedded within it using the
``-fembed-offload-object`` flag in Clang. Given an input file containing the
magic section we can pass it to this tool to extract the data contained at that
section and run a device linking job on it.

.. code-block:: console

  clang-linker-wrapper -host-triple x86_64-unknown-linux-gnu -linker-path /usr/bin/ld -- <Args>
