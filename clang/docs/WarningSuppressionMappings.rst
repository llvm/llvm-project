============================
Warning suppression mappings
============================

.. contents::
   :local:

Introduction
============

Warning suppression mappings enables users to suppress clang's diagnostics in a
per-file granular manner. Enabling enforcement of diagnostics in specific parts
of the project, even if there are violations in dependencies or other parts of
the codebase.

Goal and usage
==============

Clang allows diagnostics to be configured at a translation-unit granularity.
If a foo.cpp is compiled with -Wfoo, all transitively included headers also need
to be clean. Hence turning on new warnings at large codebases is quite difficult
today:
- It requires cleaning up all the existing warnings, which might not be possible
  when some dependencies aren't in project owner's control.
- Preventing backsliding in the meanwhile as the diagnostic can't be enforced at
  all until codebase is cleaned up.

Warning suppression mappings aims to alleviate some of these concerns by making
diagnostic configuration granularity finer, at a source file level.

To achieve this, user may create a file listing which diagnostic groups to
suppress in which files, and pass it as a command line argument to clang with
``--warning-suppression-mappings`` flag.

Note that this mechanism won't enable any diagnostics on its own. Users should
still turn on warnings in their compilations with explicit ``-Wfoo`` flags.

Example
=======

.. code-block:: bash

  $ cat my/user/code.cpp
  #include <foo/bar.h>
  namespace { void unused_func1(); }

  $ cat foo/bar.h
  namespace { void unused_func2(); }

  $ cat suppression_mappings.txt
  # Suppress -Wunused warnings in all files, apart from the ones under `foo/`.
  [unused]
  src:*
  src:*foo/*=emit
  $ clang -Wunused --warning-suppression-mappings=suppression_mappings.txt my/user/code.cpp
  # prints warning: unused function 'unused_func2', but no warnings for `unused_func1`.

Format
======

Warning suppression mappings uses a format similar to
:doc:`SanitizerSpecialCaseList`.

Users can mention sections to describe which diagnostic group behaviours to
change. Sections are denoted as ``[unused]`` in this format. Each section name
must match a diagnostic group.
When a diagnostic is matched by multiple groups, the latest one takes
precendence.

Afterwards in each section, users can have multiple entities that match source
files based on the globs. These entities look like ``src:*/my/dir/*``.
Users can also use ``emit`` category to exclude a subdirectory from suppression.
Source files are matched against these globs either as paths relative to current
working directory, or as absolute paths.
When a source file matches multiple globs, the longest one takes precendence.

.. code-block:: bash

    # Lines starting with # are ignored.
    # Configure suppression globs for `-Wunused` warnings
    [unused]
    # Suppress on all files by default.
    src:*
    # But enforce for all the sources under foo/.
    src:*foo/*=emit

    # unused-function warnings are a subgroup of `-Wunused`. So this section
    # takes precedence over the previous one for unused-function warnings, but
    # not for unused-variable warnings.
    [unused-function]
    # Only suppress for sources under bar/.
    src:*bar/*
