==========
Flang-Tidy
==========

.. contents::

See also:

.. toctree::
   :maxdepth: 1

   The list of flang-tidy checks <checks/list>

:program:`flang-tidy` is a flang-based Fortran "linter" tool. Its purpose is to
provide an extensible framework for diagnosing and fixing typical programming
errors, like style violations, interface misuse, or bugs that can be deduced via
static analysis. :program:`flang-tidy` is modular and provides a convenient
interface for writing new checks.


Using flang-tidy
================

:program:`flang-tidy` is a flang-based tool, and it's easier to work with if you
set up a compile command database for your project. You can also specify
compilation options on the command line after ``--``:

.. code-block:: console

  $ flang-tidy test.cpp -- -Imy_project/include -DMY_DEFINES ...

:program:`flang-tidy` has its own checks. Each check has a name and the checks
to run can be chosen using the ``-checks=`` option, which specifies a
comma-separated list of positive and negative (prefixed with ``-``)
globs. Positive globs add subsets of checks, and negative globs remove them. For
example,

.. code-block:: console

  $ flang-tidy test.cpp -checks=-*,bugprone-*,-bugprone-arithmetic-*

will disable all default checks (``-*``) and enable all ``bugprone-*``
checks except for ``bugprone-arithmetic-*`` ones.

The ``-list-checks`` option lists all the enabled checks. When used without
``-checks=``, it shows checks enabled by default. Use ``-checks=*`` to see all
available checks or with any other value of ``-checks=`` to see which checks are
enabled by this value.

There are currently the following groups of checks:

====================== =========================================================
Name prefix            Description
====================== =========================================================
``bugprone-``          Checks that target bug-prone code constructs.
``modernize-``         Checks that advocate usage of modern language constructs.
``openmp-``            Checks that target OpenMP constructs.
``performance-``       Checks that target performance-related issues.
``readability-``       Checks that target readability-related issues.
====================== =========================================================

Flang diagnostics are not treated by :program:`flang-tidy` and can currently not
be filtered out using the ``-checks=`` option.


Suppressing Undesired Diagnostics
=================================

:program:`flang-tidy` diagnostics are intended to call out code that does not
adhere to a coding standard, or is otherwise problematic in some way. However,
if the code is known to be correct, it may be useful to silence the warning.

If a specific suppression mechanism is not available for a certain warning, or
its use is not desired for some reason, :program:`flang-tidy` has a generic
mechanism to suppress diagnostics using ``NOLINT``.

The ``NOLINT`` comment instructs :program:`flang-tidy` to ignore warnings on the
*same line* (it doesn't apply to a function, a block of code or any other
language construct; it applies to the line of code it is on).
