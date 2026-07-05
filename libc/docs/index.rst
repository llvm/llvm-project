==================
The LLVM C Library
==================

LLVM-libc is a from-scratch implementation of the C standard library, built as
part of the LLVM project.  It is designed to be **modular** (any piece can be
used independently), **multiplatform** (Linux, GPU, baremetal embedded, UEFI,
macOS, Windows), and written in modern C++ for correctness, performance, and
safety.

What Works Today
================

LLVM-libc is **actively used in production** for a targeted set of use cases,
though coverage is still growing and many programs that depend on the full C
standard library (regex, locale, wide-character I/O, etc.) will not yet compile
against it:

* **Static-linked Linux servers and containers** — used in production at Google
  (servers and Pixel Buds) on x86-64 and AArch64.
* **GPU compute (AMDGPU, NVPTX)** — libc functions available in GPU kernels
  via LLVM's offloading runtime.  :doc:`GPU docs <gpu/index>`
* **Baremetal embedded (ARM, RISC-V, AArch64)** — minimal footprint builds
  for microcontrollers and custom hardware.
* **UEFI applications** — experimental support for firmware development.
  :doc:`UEFI docs <uefi/index>`
* **LLVM ecosystem internals** — libc++ and the offloading runtime consume
  LLVM-libc directly via :doc:`Hand-in-Hand <hand_in_hand>`.
* **Toolchain integrations** — pieces of LLVM-libc are used in Android Bionic,
  Fuchsia, Emscripten, and the ARM embedded toolchain.

Coverage is still growing.  See the :doc:`implementation status <headers/index>`
pages for per-header detail, and the
:doc:`platform support <platform_support>` page for OS/architecture coverage.

Getting Started
===============

If you are new to LLVM-libc, :doc:`getting_started` is the right first stop.
It covers cloning, building, testing, and verifying your installation in one
place.

Want to use LLVM-libc *alongside* your system libc instead of replacing it?
See :doc:`overlay_mode`.

Get Involved
============

LLVM-libc is an active project and welcomes contributors of all experience
levels.  See :doc:`contributing` to learn how to help.

* `Source code <https://github.com/llvm/llvm-project/tree/main/libc>`__
* `Bug reports <https://github.com/llvm/llvm-project/labels/libc>`__
* `Discourse <https://discourse.llvm.org/c/runtimes/libc>`__
* `Discord <https://discord.com/channels/636084430946959380/636732994891284500>`__
  (`invite <https://discord.gg/xS7Z362>`__)
* `Buildbot <https://lab.llvm.org/buildbot/#/builders?tags=libc>`__

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Using LLVM-libc

   getting_started
   build_concepts
   overlay_mode
   full_host_build
   full_cross_build
   configure
   hand_in_hand

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Platforms

   gpu/index.rst
   uefi/index.rst

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Implementation Status

   headers/index.rst
   arch_support
   platform_support
   compiler_support

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Development

   contributing
   build_and_test
   dev/index.rst
   porting
   Maintainers

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Links

   talks
   Source Code <https://github.com/llvm/llvm-project/tree/main/libc>
   Bug Reports <https://github.com/llvm/llvm-project/labels/libc>
   Discourse <https://discourse.llvm.org/c/runtimes/libc>
   Join the Discord <https://discord.gg/xS7Z362>
   Discord Channel <https://discord.com/channels/636084430946959380/636732994891284500>
   Buildbot <https://lab.llvm.org/buildbot/#/builders?tags=libc>
