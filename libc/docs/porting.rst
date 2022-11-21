.. _porting:

=======================================
Bringup on a New OS or Architecture
=======================================

.. contents:: Table of Contents
  :depth: 4
  :local:

CI builders
===========

If you are contributing a port for a operating system or architecture which
is not covered by existing CI builders, you will also have to present a plan
for testing and contribute a CI builder. See
`this guide <https://llvm.org/docs/HowToAddABuilder.html>`_ for information
on how to add new builders to the
`LLVM buildbot <https://lab.llvm.org/buildbot>`_.
You will either have to extend the existing
`Linux script <https://github.com/llvm/llvm-zorg/blob/main/zorg/buildbot/builders/annotated/libc-linux.py>`_
and/or
`Windows script <https://github.com/llvm/llvm-zorg/blob/main/zorg/buildbot/builders/annotated/libc-windows.py>`_
or add a new script for your operating system.

An OS specific config directory
===============================

If you are starting to bring up LLVM's libc on a new operating system, the first
step is to add a directory for that OS in the ``libc/config`` directory. Both
`Linux <https://github.com/llvm/llvm-project/tree/main/libc/config/linux>`_ and
`Windows <https://github.com/llvm/llvm-project/tree/main/libc/config/windows>`_,
the two operating systems on which LLVM's libc is being actively developed, 
have their own config directory.

.. note:: Windows development is not as active as the development on Linux.
   There is a
   `Darwin <https://github.com/llvm/llvm-project/tree/main/libc/config/darwin>`_
   config also which is in a similar state as Windows.

.. note:: LLVM's libc is being brought up on the
   `Fuchsia <https://fuchsia.dev/>`_ operating system also. However, there is no
   config directory for Fuchsia as the bring up is being done in the Fuchsia
   source tree.

The api.td file
---------------

If the :ref:`fullbuild_mode` is to be supported on the new operating system,
then a file named ``api.td`` should be added in its config directory. It is
written in the
`LLVM tablegen language <https://llvm.org/docs/TableGen/ProgRef.html>`_.
It lists all the relevant macros and type definitions we want in the
public libc header files. See the existing Linux
`api.td <https://github.com/llvm/llvm-project/blob/main/libc/config/linux/api.td>`_
file as an example to prepare the ``api.td`` file for the new operating system.

.. note:: In future, LLVM tablegen will be replaced with a different DSL to list
   config information.

Architecture Subdirectory
=========================

There are parts of the libc which are implemented differently for different
architectures. The simplest example of this is the ``syscall`` function and
its internal implementation - its Linux implementation differs for different
architectures. Since a large part of the libc makes use of syscalls (or an
equivalent on non-Linux like platforms), it might be simpler and convenient to
bring up the libc for one architecture at a time. In such cases, wherein the
support surface of LLVM's libc differs for each target architecture, one will
have to add a subdirectory (within the config directory os the operating
system) for each target architecture, and list the relevant config information
separately in those subdirectories. For example, for Linux, the x86_64 and
aarch64 configs are in separate directories, named
`x86_64 <https://github.com/llvm/llvm-project/tree/main/libc/config/linux/x86_64>`_
and `aarch64 <https://github.com/llvm/llvm-project/tree/main/libc/config/linux/aarch64>`_.
The libc CMake machinery looks for subdirectories named after the target
architecture.

The entrypoints.txt file
========================

One of the important pieces of config information is listed in a file named
``entrypoints.txt``. This file lists the targets for the entrypoints (see
:ref:`entrypoints`) you want to include in the static archive of the libc (for
the :ref:`overlay_mode` and/or the :ref:`fullbuild_mode`.) If you are doing an
architecture specific bring up, then an ``entrypoints.txt`` file should be
created in the architecture subdirectory for each architecture. Else, having a
single ``entrypoints.txt`` in the operating system directory is sufficient.

The Linux config has an ``entrypoint.txt`` for each individual target
architecture separately: `aarch64 <https://github.com/llvm/llvm-project/tree/main/libc/config/linux/aarch64>`_,
`arm32 <https://github.com/llvm/llvm-project/tree/main/libc/config/linux/arm>`_ and
`x86_64 <https://github.com/llvm/llvm-project/tree/main/libc/config/linux/x86_64>`_. On the
other hand, the Windows config has a single ``entrypoints.txt``
`file <https://github.com/llvm/llvm-project/tree/main/libc/config/windows/entrypoints.txt>`_.

A typical bring up procedure will normally bring up a small group of entrypoints
at a time. The usual practice is to progressively add the targets for those
entrypoints to the ``entrypoints.txt`` file as they are being brought up. The
same is the case if one is implementing a new entrypoint - the target for the
new entrypoint should be added to the relevant ``entrypoints.txt`` file. If
the implementation of the new entrypoint supports multiple operating systems and
target architectures, then multiple ``entrypoints.txt`` files will have to be
updated.

The headers.txt file
====================

Another important piece of config informtion is listed in a file named
``headers.txt``. It lists the targets for the set of public headers that are
provided by the libc. This is relevant only if the libc is to be used in the
:ref:`fullbuild_mode` on the target operating system and architecture. As with
the ``entrypoints.txt`` file, one ``headers.txt`` file should be listed for
each individual target architecture if you are doing an architecture specific
bring up. The Linux config has ``headers.txt`` file listed seperately for the
`aarch64 <https://github.com/llvm/llvm-project/tree/main/libc/config/linux/aarch64>`_
config and the
`x86_64 <https://github.com/llvm/llvm-project/tree/main/libc/config/linux/x86_64>`_
config.
