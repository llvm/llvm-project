.. If you want to modify sections/contents permanently, you should modify both
   ReleaseNotes.rst and ReleaseNotesTemplate.txt.

===========================
lld |release| Release Notes
===========================

.. contents::
    :local:

.. only:: PreRelease

  .. warning::
     These are in-progress notes for the upcoming LLVM |release| release.
     Release notes for previous releases can be found on
     `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the lld linker, release |release|.
Here we describe the status of lld, including major improvements
from the previous release. All lld releases may be downloaded
from the `LLVM releases web site <https://llvm.org/releases/>`_.

Non-comprehensive list of changes in this release
=================================================

ELF Improvements
----------------

* Added ``--bp-compression-sort-section=<glob>[=<layout_priority>[=<match_priority>]]``,
  replacing the old coarse ``--bp-compression-sort`` modes with a way to split
  input sections into multiple compression groups, run balanced partitioning
  independently per group, and leave out sections that are poor candidates for
  BP.
  ``layout_priority`` controls group placement order (lower value = placed
  first, default 0). ``match_priority`` resolves conflicts when multiple globs
  match the same section (lower value = higher priority; explicit priority
  beats positional last-match-wins; default: positional). In ELF, the glob
  matches input section names (e.g. ``.text.unlikely.code1``).

Breaking changes
----------------

COFF Improvements
-----------------

MinGW Improvements
------------------

* Added ``--push-state`` and ``--pop-state``, offering the same semantics as
  when used with the ELF linker: The state of ``--Bstatic``/``--Bdynamic`` and
  ``--whole-archive`` are pushed onto a stack and popped from it.

MachO Improvements
------------------

* ``--bp-compression-sort-section`` now accepts optional layout and match
  priorities (same syntax as ELF). In Mach-O, the glob matches the
  concatenated segment+section name (e.g. ``__TEXT__text``).

WebAssembly Improvements
------------------------

Fixes
#####
