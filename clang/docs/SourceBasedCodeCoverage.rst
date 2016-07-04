==========================
Source-based Code Coverage
==========================

.. contents::
   :local:

Introduction
============

This document explains how to use clang's source-based code coverage feature.
It's called "source-based" because it operates on AST and preprocessor
information directly. This allows it to generate very precise coverage data.

Clang ships two other code coverage implementations:

* :doc:`SanitizerCoverage` - A low-overhead tool meant for use alongside the
  various sanitizers. It can provide up to edge-level coverage.

* gcov - A GCC-compatible coverage implementation which operates on DebugInfo.

From this point onwards "code coverage" will refer to the source-based kind.

The code coverage workflow
==========================

The code coverage workflow consists of three main steps:

* Compiling with coverage enabled.

* Running the instrumented program.

* Creating coverage reports.

The next few sections work through a complete, copy-'n-paste friendly example
based on this program:

.. code-block:: cpp

    % cat <<EOF > foo.cc
    #define BAR(x) ((x) || (x))
    template <typename T> void foo(T x) {
      for (unsigned I = 0; I < 10; ++I) { BAR(I); }
    }
    int main() {
      foo<int>(0);
      foo<float>(0);
      return 0;
    }
    EOF

Compiling with coverage enabled
===============================

To compile code with coverage enabled, pass ``-fprofile-instr-generate
-fcoverage-mapping`` to the compiler:

.. code-block:: console

    # Step 1: Compile with coverage enabled.
    % clang++ -fprofile-instr-generate -fcoverage-mapping foo.cc -o foo

Note that linking together code with and without coverage instrumentation is
supported: any uninstrumented code simply won't be accounted for.

Running the instrumented program
================================

The next step is to run the instrumented program. When the program exits it
will write a **raw profile** to the path specified by the ``LLVM_PROFILE_FILE``
environment variable. If that variable does not exist, the profile is written
to ``default.profraw`` in the current directory of the program. If
``LLVM_PROFILE_FILE`` contains a path to a non-existent directory, the missing
directory structure will be created.  Additionally, the following special
**pattern strings** are rewritten:

* "%p" expands out to the process ID.

* "%h" expands out to the hostname of the machine running the program.

* "%Nm" expands out to the instrumented binary's signature. When this pattern
  is specified, the runtime creates a pool of N raw profiles which are used for
  on-line profile merging. The runtime takes care of selecting a raw profile
  from the pool, locking it, and updating it before the program exits.  If N is
  not specified (i.e the pattern is "%m"), it's assumed that ``N = 1``. N must
  be between 1 and 9. The merge pool specifier can only occur once per filename
  pattern.

.. code-block:: console

    # Step 2: Run the program.
    % LLVM_PROFILE_FILE="foo.profraw" ./foo

Creating coverage reports
=========================

Raw profiles have to be **indexed** before they can be used to generate
coverage reports. This is done using the "merge" tool in ``llvm-profdata``, so
named because it can combine and index profiles at the same time:

.. code-block:: console

    # Step 3(a): Index the raw profile.
    % llvm-profdata merge -sparse foo.profraw -o foo.profdata

There are multiple different ways to render coverage reports. One option is to
generate a line-oriented report:

.. code-block:: console

    # Step 3(b): Create a line-oriented coverage report.
    % llvm-cov show ./foo -instr-profile=foo.profdata

To demangle any C++ identifiers in the output, use:

.. code-block:: console

    % llvm-cov show ./foo -instr-profile=foo.profdata | c++filt -n

This report includes a summary view as well as dedicated sub-views for
templated functions and their instantiations. For our example program, we get
distinct views for ``foo<int>(...)`` and ``foo<float>(...)``.  If
``-show-line-counts-or-regions`` is enabled, ``llvm-cov`` displays sub-line
region counts (even in macro expansions):

.. code-block:: none

       20|    1|#define BAR(x) ((x) || (x))
                               ^20     ^2
        2|    2|template <typename T> void foo(T x) {
       22|    3|  for (unsigned I = 0; I < 10; ++I) { BAR(I); }
                                       ^22     ^20  ^20^20
        2|    4|}
    ------------------
    | void foo<int>(int):
    |      1|    2|template <typename T> void foo(T x) {
    |     11|    3|  for (unsigned I = 0; I < 10; ++I) { BAR(I); }
    |                                     ^11     ^10  ^10^10
    |      1|    4|}
    ------------------
    | void foo<float>(int):
    |      1|    2|template <typename T> void foo(T x) {
    |     11|    3|  for (unsigned I = 0; I < 10; ++I) { BAR(I); }
    |                                     ^11     ^10  ^10^10
    |      1|    4|}
    ------------------

It's possible to generate a file-level summary of coverage statistics (instead
of a line-oriented report) with:

.. code-block:: console

    # Step 3(c): Create a coverage summary.
    % llvm-cov report ./foo -instr-profile=foo.profdata
    Filename                    Regions    Miss   Cover Functions  Executed
    -----------------------------------------------------------------------
    /tmp/foo.cc                      13       0 100.00%         3   100.00%
    -----------------------------------------------------------------------
    TOTAL                            13       0 100.00%         3   100.00%

A few final notes:

* The ``-sparse`` flag is optional but can result in dramatically smaller
  indexed profiles. This option should not be used if the indexed profile will
  be reused for PGO.

* Raw profiles can be discarded after they are indexed. Advanced use of the
  profile runtime library allows an instrumented program to merge profiling
  information directly into an existing raw profile on disk. The details are
  out of scope.

* The ``llvm-profdata`` tool can be used to merge together multiple raw or
  indexed profiles. To combine profiling data from multiple runs of a program,
  try e.g:

  .. code-block:: console

      % llvm-profdata merge -sparse foo1.profraw foo2.profdata -o foo3.profdata

Format compatibility guarantees
===============================

* There are no backwards or forwards compatibility guarantees for the raw
  profile format. Raw profiles may be dependent on the specific compiler
  revision used to generate them. It's inadvisable to store raw profiles for
  long periods of time.

* Tools must retain **backwards** compatibility with indexed profile formats.
  These formats are not forwards-compatible: i.e, a tool which uses format
  version X will not be able to understand format version (X+k).

* There is a third format in play: the format of the coverage mappings emitted
  into instrumented binaries. Tools must retain **backwards** compatibility
  with these formats. These formats are not forwards-compatible.

Using the profiling runtime without static initializers
=======================================================

By default the compiler runtime uses a static initializer to determine the
profile output path and to register a writer function. To collect profiles
without using static initializers, do this manually:

* Export a ``int __llvm_profile_runtime`` symbol from each instrumented shared
  library and executable. When the linker finds a definition of this symbol, it
  knows to skip loading the object which contains the profiling runtime's
  static initializer.

* Forward-declare ``void __llvm_profile_initialize_file(void)`` and call it
  once from each instrumented executable. This function parses
  ``LLVM_PROFILE_FILE``, sets the output path, and truncates any existing files
  at that path. To get the same behavior without truncating existing files,
  pass a filename pattern string to ``void __llvm_profile_set_filename(char
  *)``.  These calls can be placed anywhere so long as they precede all calls
  to ``__llvm_profile_write_file``.

* Forward-declare ``int __llvm_profile_write_file(void)`` and call it to write
  out a profile. This function returns 0 when it succeeds, and a non-zero value
  otherwise. Calling this function multiple times appends profile data to an
  existing on-disk raw profile.

Drawbacks and limitations
=========================

* Code coverage does not handle unpredictable changes in control flow or stack
  unwinding in the presence of exceptions precisely. Consider the following
  function:

  .. code-block:: cpp

      int f() {
        may_throw();
        return 0;
      }

  If the call to ``may_throw()`` propagates an exception into ``f``, the code
  coverage tool may mark the ``return`` statement as executed even though it is
  not. A call to ``longjmp()`` can have similar effects.
