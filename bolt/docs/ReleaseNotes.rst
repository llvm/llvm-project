===========================
openmp 11.0.0 Release Notes
===========================

.. contents::
    :local:

.. warning::
   These are in-progress notes for the upcoming LLVM 11.0.0 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the OpenMP runtime, release 11.0.0.
Here we describe the status of openmp, including major improvements
from the previous release. All openmp releases may be downloaded
from the `LLVM releases web site <https://llvm.org/releases/>`_.

Non-comprehensive list of changes in this release
=================================================

5.0 features
------------

* ...

5.1 features
------------

* ...

OMPT Improvements
-----------------

* Added OMPT callbacks for doacross loops, detached tasks
* Added handling for mutexinoutset dependencies

OMPT-based Tools
----------------

* Added ompt-multiplex.h as a header-only OMPT-tool to support nesting of OMPT
  tools. (see openmp/tools/multiplex)

