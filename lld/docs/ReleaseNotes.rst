========================
lld 13.0.0 Release Notes
========================

.. contents::
    :local:

.. warning::
   These are in-progress notes for the upcoming LLVM 13.0.0 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the lld linker, release 13.0.0.
Here we describe the status of lld, including major improvements
from the previous release. All lld releases may be downloaded
from the `LLVM releases web site <https://llvm.org/releases/>`_.

Non-comprehensive list of changes in this release
=================================================

ELF Improvements
----------------

* ``-Bsymbolic -Bsymbolic-functions`` has been changed to behave the same as ``-Bsymbolic-functions``. This matches GNU ld.
  (`D102461 <https://reviews.llvm.org/D102461>`_)
* ``-Bno-symbolic`` has been added.
  (`D102461 <https://reviews.llvm.org/D102461>`_)
* A new linker script command ``OVERWRITE_SECTIONS`` has been added.
  (`D103303 <https://reviews.llvm.org/D103303>`_)
* ``-Bsymbolic-non-weak-functions`` has been added as a ``STB_GLOBAL`` subset of ``-Bsymbolic-functions``.
  (`D102570 <https://reviews.llvm.org/D102570>`_)

Breaking changes
----------------

* ``--shuffle-sections=<seed>`` has been changed to ``--shuffle-sections=<section-glob>=<seed>``.
  Specify ``*`` as ``<section-glob>`` to get the previous behavior.

COFF Improvements
-----------------

* Avoid thread exhaustion when running on 32 bit Windows.
  (`D105506 <https://reviews.llvm.org/D105506>`_)

* Improve terminating the process on Windows while a thread pool might be
  running. (`D102944 <https://reviews.llvm.org/D102944>`_)

MinGW Improvements
------------------

* Support for linking directly against a DLL without using an import library
  has been added. (`D104530 <https://reviews.llvm.org/D104530>`_ and
  `D104531 <https://reviews.llvm.org/D104531>`_)

* Fix linking with ``--export-all-symbols`` in combination with
  ``-function-sections``. (`D101522 <https://reviews.llvm.org/D101522>`_ and
  `D101615 <https://reviews.llvm.org/D101615>`_)

* Fix automatic export of symbols from LTO objects.
  (`D101569 <https://reviews.llvm.org/D101569>`_)

* Accept more spellings of some options.
  (`D107237 <https://reviews.llvm.org/D107237>`_ and
  `D107253 <https://reviews.llvm.org/D107253>`_)

Mach-O Improvements
-------------------

The Mach-O backend is now able to link several large, real-world programs,
though we are still working out the kinks.

* arm64 is now supported as a target. (`D88629 <https://reviews.llvm.org/D88629>`_)
* arm64_32 is now supported as a target. (`D99822 <https://reviews.llvm.org/D99822>`_)
* Branch-range-extension thunks are now supported. (`D100818 <https://reviews.llvm.org/D100818>`_)
* ``-dead_strip`` is now supported. (`D103324 <https://reviews.llvm.org/D103324>`_)
* Support for identical code folding (``--icf=all``) has been added.
  (`D103292 <https://reviews.llvm.org/D103292>`_)
* Support for special ``$start`` and ``$end`` symbols for segment & sections has been
  added. (`D106767 <https://reviews.llvm.org/D106767>`_, `D106629 <https://reviews.llvm.org/D106629>`_)
* ``$ld$previous`` symbols are now supported. (`D103505 <https://reviews.llvm.org/D103505 >`_)
* ``$ld$install_name`` symbols are now supported. (`D103746 <https://reviews.llvm.org/D103746>`_)
* ``__mh_*_header`` symbols are now supported. (`D97007 <https://reviews.llvm.org/D97007>`_)
* LC_CODE_SIGNATURE is now supported. (`D96164 <https://reviews.llvm.org/D96164>`_)
* LC_FUNCTION_STARTS is now supported. (`D97260 <https://reviews.llvm.org/D97260>`_)
* LC_DATA_IN_CODE is now supported. (`D103006 <https://reviews.llvm.org/D103006>`_)
* Bind opcodes are more compactly encoded. (`D106128 <https://reviews.llvm.org/D106128>`_,
  `D105075 <https://reviews.llvm.org/D105075>`_)
* LTO cache support has been added. (`D105922 <https://reviews.llvm.org/D105922>`_)
* ``-application_extension`` is now supported. (`D105818 <https://reviews.llvm.org/D105818>`_)
* ``-export_dynamic`` is now partially supported. (`D105482 <https://reviews.llvm.org/D105482>`_)
* ``-arch_multiple`` is now supported. (`D105450 <https://reviews.llvm.org/D105450>`_)
* ``-final_output`` is now supported. (`D105449 <https://reviews.llvm.org/D105449>`_)
* ``-umbrella`` is now supported. (`D105448 <https://reviews.llvm.org/D105448>`_)
* ``--print-dylib-search`` is now supported. (`D103985 <https://reviews.llvm.org/D103985>`_)
* ``-force_load_swift_libs`` is now supported. (`D103709 <https://reviews.llvm.org/D103709>`_)
* ``-reexport_framework``, ``-reexport_library``, ``-reexport-l`` are now supported.
  (`D103497 <https://reviews.llvm.org/D103497>`_)
* ``.weak_def_can_be_hidden`` is now supported. (`D101080 <https://reviews.llvm.org/D101080>`_)
* ``-add_ast_path`` is now supported. (`D100076 <https://reviews.llvm.org/D100076>`_)
* ``-segprot`` is now supported.  (`D99389 <https://reviews.llvm.org/D99389>`_)
* ``-dependency_info`` is now partially supported. (`D98559 <https://reviews.llvm.org/D98559>`_)
* ``--time-trace`` is now supported. (`D98419 <https://reviews.llvm.org/D98419>`_)
* ``-mark_dead_strippable_dylib`` is now supported. (`D98262 <https://reviews.llvm.org/D98262>`_)
* ``-[un]exported_symbol[s_list]`` is now supported. (`D98223 <https://reviews.llvm.org/D98223>`_)
* ``-flat_namespace`` is now supported. (`D97641 <https://reviews.llvm.org/D97641>`_)
* ``-rename_section`` and ``-rename_segment`` are now supported. (`D97600 <https://reviews.llvm.org/D97600>`_)
* ``-bundle_loader`` is now supported. (`D95913 <https://reviews.llvm.org/D95913>`_)
* ``-map`` is now partially supported. (`D98323 <https://reviews.llvm.org/D98323>`_)

There were numerous other bug-fixes as well.

WebAssembly Improvements
------------------------

