.. _porting:

=======================================
Bringup on a New OS or Architecture
=======================================

Building the libc
=================

An OS specific config directory
-------------------------------

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

Architecture Subdirectory
-------------------------

There are parts of the libc which are implemented differently for different
architectures. The simplest example of this is the ``syscall`` function and
its internal implementation - its Linux implementation differs for different
architectures. Since a large part of the libc makes use of syscalls (or an
equivalent on non-Linux like platforms), it might be simpler and convenient to
bring up the libc for one architecture at a time. In such cases, wherein the
support surface of LLVM's libc differs for each target architecture, one will
have to add a subdirectory (within the config directory of the operating
system) for each target architecture, and list the relevant config information
separately in those subdirectories. For example, for Linux, the x86_64 and
aarch64 configs are in separate directories, named
`x86_64 <https://github.com/llvm/llvm-project/tree/main/libc/config/linux/x86_64>`_
and `aarch64 <https://github.com/llvm/llvm-project/tree/main/libc/config/linux/aarch64>`_.
The libc CMake machinery looks for subdirectories named after the target
architecture.

The entrypoints.txt file
------------------------

The ``entrypoints.txt`` file lists the targets for the entrypoints to be
included in the build for a specific platform. For more technical details on
what entrypoints are and how they are registered as targets, see the
:ref:`entrypoints` documentation.

If you are doing an architecture specific bring-up, then an ``entrypoints.txt``
file should be created in the architecture subdirectory for each architecture.
Else, having a single ``entrypoints.txt`` in the operating system directory is
sufficient.

A typical bring-up procedure will normally involve progressively adding targets
to this file as they are implemented and tested.


The headers.txt file
--------------------

Another important piece of config information is listed in a file named
``headers.txt``. It lists the targets for the set of public headers that are
provided by the libc. This is relevant only if the libc is to be used in the
:ref:`full_host_build` on the target operating system and architecture. As with
the ``entrypoints.txt`` file, one ``headers.txt`` file should be listed for
each individual target architecture if you are doing an architecture specific
bring up. The Linux config has ``headers.txt`` file listed separately for the
`aarch64 <https://github.com/llvm/llvm-project/tree/main/libc/config/linux/aarch64>`_
config and the
`x86_64 <https://github.com/llvm/llvm-project/tree/main/libc/config/linux/x86_64>`_
config.


Upstreaming
===========

Adding a target to the main LLVM-libc has some requirements to ensure that the
targets stay in usable condition. LLVM-libc is under active development and
without active maintenance targets will become stale and may be sunset.

Maintenance
-----------

To add a target there must be one or more people whose responsibility it is to
keep the target up to date or push it forwards if it's not complete. Those
people are the maintainers, and they are responsible for keeping their target in
good shape. This means fixing their target when it breaks, reviewing patches
related to their target, and keeping the target's CI running.

Maintainers are listed in libc/maintainers.rst and must follow
`LLVM's maintainer policy <https://llvm.org/docs/DeveloperPolicy.html#maintainers>`_.

CI builders
-----------

Every target needs at least one CI builder. These are used to check when the
target breaks, and to help people who don't have access to the specific
architecture fix their bugs. LLVM-libc has both presubmit CI on github
and postsubmit CI on the `LLVM buildbot <https://lab.llvm.org/buildbot>`_. For
instructions on contributing a postsubmit buildbot read
`the LLVM documentation <https://llvm.org/docs/HowToAddABuilder.html>`_ and for
presubmit tests read
`the best practices <https://llvm.org/docs/CIBestPractices.html>`_

The test configurations are at these links:
 * `Linux Postsubmit <https://github.com/llvm/llvm-zorg/blob/main/zorg/buildbot/builders/annotated/libc-linux.py>`_
 * `Windows Postsubmit <https://github.com/llvm/llvm-zorg/blob/main/zorg/buildbot/builders/annotated/libc-windows.py>`_
 * `Fullbuild Presubmit <https://github.com/llvm/llvm-project/blob/main/.github/workflows/libc-fullbuild-tests.yml>`_
 * `Overlay Presubmit <https://github.com/llvm/llvm-project/blob/main/.github/workflows/libc-overlay-tests.yml>`_

Sunsetting
----------

Sunsetting is the process through which targets can be removed from LLVM-libc.
If a target is broken or stale it may be sunset. It is the responsibility of the
target's maintainers to keep it from being sunset. The target's maintainers may
be marked inactive if their target is sunset.

Broken targets are ones where the target's CI has been failing for at least
30 days. After 30 days have passed the CI should be changed so it does not
block commits and does not notify people when it fails. If the target remains
broken for another 90 days it may be sunset.

Stale targets are ones with no active maintainers or no contributions. If a
target has 0 specific contributions between two major LLVM releases, or if it
has no active maintainers, then it may be marked "deprecated" in the next major
release. If there are still no contributions or no maintainers after the next
major release it may be sunset.

To sunset a target, all specific references to that target in the code and build
system should be removed. All buildbots for that target should be shut down, or
otherwise removed.

To restart a target that was previously sunset, the new maintainers are
encouraged to look at the commit(s) removing the target to provide a starting
point.
