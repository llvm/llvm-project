======================
Clang Offload Packager
======================

.. contents::
   :local:

.. _clang-offload-packager:

Introduction
============

This tool bundles device files into a single image containing necessary
metadata. We use a custom binary format for bundling all the device images
together. The image format is a small header wrapping around a string map. This
tool creates bundled binaries so that they can be embedded into the host to
create a fat-binary.

An embedded binary is marked by the ``0x10FF10AD`` magic bytes, followed by a
version. Each created binary contains its own magic bytes. This allows us to
locate all the embedded offloading sections even after they may have been merged
by the linker, such as when using relocatable linking. The format used is
primarily a binary serialization of the following struct.

.. code-block:: c++

  struct OffloadingImage {
    uint16_t TheImageKind;
    uint16_t TheOffloadKind;
    uint32_t Flags;
    StringMap<StringRef> StringData;
    MemoryBufferRef Image;
  };

Usage
=====

This tool can be used with the following arguments. Generally information is
passed as a key-value pair to the ``image=`` argument. The ``file``, ``triple``,
and ``arch`` arguments are considered mandatory to make a valid image.

.. code-block:: console

  OVERVIEW: A utility for bundling several object files into a single binary.
  The output binary can then be embedded into the host section table
  to create a fatbinary containing offloading code.
  
  USAGE: clang-offload-packager [options]
  
  OPTIONS:
  
  Generic Options:
  
    --help                      - Display available options (--help-hidden for more)
    --help-list                 - Display list of available options (--help-list-hidden for more)
    --version                   - Display the version of this program
  
  clang-offload-packager options:
  
    --image=<<key>=<value>,...> - List of key and value arguments. Required
                                  keywords are 'file' and 'triple'.
    -o=<file>                   - Write output to <file>.

Example
=======

This tool simply takes many input files from the ``image`` option and creates a
single output file with all the images combined.

.. code-block:: console

  clang-offload-packager -o out.bin --image=file=input.o,triple=nvptx64,arch=sm_70
