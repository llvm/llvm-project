============================
Warning suppression mappings
============================

.. contents::
   :local:

Introduction
============

Warning suppression mappings enable users to suppress Clang's diagnostics at a
per-file granularity. This allows enforcing diagnostics in specific parts of the
project even if there are violations in some headers.

Goal and usage
==============

Clang allows diagnostics to be configured at a translation-unit granularity.
If a ``foo.cpp`` is compiled with ``-Wfoo``, all transitively included headers
also need to be clean. Hence, turning on new warnings in large codebases
requires cleaning up all the existing warnings. This might not be possible when
some dependencies aren't in the project owner's control or because new
violations are creeping up quicker than the clean up.

Warning suppression mappings aim to alleviate some of these concerns by making
diagnostic configuration granularity finer, at a source file level.

To achieve this, user can create a file that lists which :doc:`diagnostic
groups <DiagnosticsReference>` to suppress in which files or paths, and pass it
as a command line argument to Clang with the ``--warning-suppression-mappings``
flag.

Note that this mechanism won't enable any diagnostics on its own. Users should
still turn on warnings in their compilations with explicit ``-Wfoo`` flags.
`Controlling diagnostics pragmas
<https://clang.llvm.org/docs/UsersManual.html#controlling-diagnostics-via-pragmas>`_
take precedence over suppression mappings. Ensuring code author's explicit
intent is always preserved.

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

Warning suppression mappings uses the same format as
:doc:`SanitizerSpecialCaseList`.

Sections describe which diagnostic group's behaviour to change, e.g.
``[unused]``. When a diagnostic is matched by multiple sections, the latest
section takes precedence.

Afterwards in each section, users can have multiple entities that match source
files based on the globs. These entities look like ``src:*/my/dir/*``.
Users can also use the ``emit`` category to exclude a subdirectory from
suppression.
Source files are matched against these globs either:

- as paths relative to the current working directory
- as absolute paths.

When a source file matches multiple globs in a section, the longest one takes
precedence.

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
