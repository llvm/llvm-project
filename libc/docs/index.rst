==================
The LLVM C Library
==================

.. note::
  LLVM-libc is not fully complete right now. Some programs may fail to build due
  to missing functions. If you would like to help us finish LLVM-libc, check
  out "`Contributing to the libc project <contributing.html>`__" in the sidebar
  or ask on `discord <https://discord.com/channels/636084430946959380/636732994891284500>`__
  (`invite link <https://discord.gg/xS7Z362>`__).

Introduction
============

LLVM-libc is an implementation of the C standard library written in C++ focused
on embodying three main principles:

- Modular
- Multiplatform
- Community Oriented

Our current goal is to support users who want to make libc part of their
application. This can be through static linking libc into the application, which
is common for containerized servers or embedded devices. It can also be through
using the LLVM-libc internal sources as a library, such as through the
:ref:`Hand-in-Hand interface<hand_in_hand>`.


TODO: Finish list of where LLVM-libc is used.
LLVM-libc is currently used in Google servers, Pixel Buds, and other Google
projects. There is an experiemental config to use LLVM-libc in Emscripten.
Pieces of LLVM-libc are being used in Bionic (Android's libc) and Fuchsia.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Status & Support

   headers/index.rst
   arch_support
   platform_support
   compiler_support

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Simple Usage

   getting_started

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Advanced Usage

   full_host_build
   full_cross_build
   overlay_mode
   gpu/index.rst
   uefi/index.rst
   configure
   hand_in_hand

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Development

   Maintainers
   build_and_test
   dev/index.rst
   porting
   contributing

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Useful Links

   talks
   Source Code <https://github.com/llvm/llvm-project/tree/main/libc>
   Bug Reports <https://github.com/llvm/llvm-project/labels/libc>
   Discourse <https://discourse.llvm.org/c/runtimes/libc>
   Join the Discord <https://discord.gg/xS7Z362>
   Discord Channel <https://discord.com/channels/636084430946959380/636732994891284500>
   Buildbot <https://lab.llvm.org/buildbot/#/builders?tags=libc>
