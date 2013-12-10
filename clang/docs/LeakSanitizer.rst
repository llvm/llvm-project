================
LeakSanitizer
================

.. contents::
   :local:

Introduction
============

LeakSanitizer is a run-time memory leak detector. It can be combined with
:doc:`AddressSanitizer` to get both memory error and leak detection.
LeakSanitizer does not introduce any additional slowdown when used in this mode.
The LeakSanitizer runtime can also be linked in separately to get leak detection
only, at a minimal performance cost.

Current status
==============

LeakSanitizer is experimental and supported only on x86\_64 Linux.

The combined mode has been tested on fairly large software projects. The
stand-alone mode has received much less testing.

There are plans to support LeakSanitizer in :doc:`MemorySanitizer` builds.

More Information
================

`https://code.google.com/p/address-sanitizer/wiki/LeakSanitizer
<https://code.google.com/p/address-sanitizer/wiki/LeakSanitizer>`_

