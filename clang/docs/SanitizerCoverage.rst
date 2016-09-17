=================
SanitizerCoverage
=================

.. contents::
   :local:

Introduction
============

Sanitizer tools have a very simple code coverage tool built in. It allows to
get function-level, basic-block-level, and edge-level coverage at a very low
cost.

How to build and run
====================

SanitizerCoverage can be used with :doc:`AddressSanitizer`,
:doc:`LeakSanitizer`, :doc:`MemorySanitizer`,
UndefinedBehaviorSanitizer, or without any sanitizer.  Pass one of the
following compile-time flags:

* ``-fsanitize-coverage=func`` for function-level coverage (very fast).
* ``-fsanitize-coverage=bb`` for basic-block-level coverage (may add up to 30%
  **extra** slowdown).
* ``-fsanitize-coverage=edge`` for edge-level coverage (up to 40% slowdown).

You may also specify ``-fsanitize-coverage=indirect-calls`` for
additional `caller-callee coverage`_.

At run time, pass ``coverage=1`` in ``ASAN_OPTIONS``,
``LSAN_OPTIONS``, ``MSAN_OPTIONS`` or ``UBSAN_OPTIONS``, as
appropriate. For the standalone coverage mode, use ``UBSAN_OPTIONS``.

To get `Coverage counters`_, add ``-fsanitize-coverage=8bit-counters``
to one of the above compile-time flags. At runtime, use
``*SAN_OPTIONS=coverage=1:coverage_counters=1``.

Example:

.. code-block:: console

    % cat -n cov.cc
         1  #include <stdio.h>
         2  __attribute__((noinline))
         3  void foo() { printf("foo\n"); }
         4
         5  int main(int argc, char **argv) {
         6    if (argc == 2)
         7      foo();
         8    printf("main\n");
         9  }
    % clang++ -g cov.cc -fsanitize=address -fsanitize-coverage=func
    % ASAN_OPTIONS=coverage=1 ./a.out; ls -l *sancov
    main
    -rw-r----- 1 kcc eng 4 Nov 27 12:21 a.out.22673.sancov
    % ASAN_OPTIONS=coverage=1 ./a.out foo ; ls -l *sancov
    foo
    main
    -rw-r----- 1 kcc eng 4 Nov 27 12:21 a.out.22673.sancov
    -rw-r----- 1 kcc eng 8 Nov 27 12:21 a.out.22679.sancov

Every time you run an executable instrumented with SanitizerCoverage
one ``*.sancov`` file is created during the process shutdown.
If the executable is dynamically linked against instrumented DSOs,
one ``*.sancov`` file will be also created for every DSO.

Postprocessing
==============

The format of ``*.sancov`` files is very simple: the first 8 bytes is the magic,
one of ``0xC0BFFFFFFFFFFF64`` and ``0xC0BFFFFFFFFFFF32``. The last byte of the
magic defines the size of the following offsets. The rest of the data is the
offsets in the corresponding binary/DSO that were executed during the run.

A simple script
``$LLVM/projects/compiler-rt/lib/sanitizer_common/scripts/sancov.py`` is
provided to dump these offsets.

.. code-block:: console

    % sancov.py print a.out.22679.sancov a.out.22673.sancov
    sancov.py: read 2 PCs from a.out.22679.sancov
    sancov.py: read 1 PCs from a.out.22673.sancov
    sancov.py: 2 files merged; 2 PCs total
    0x465250
    0x4652a0

You can then filter the output of ``sancov.py`` through ``addr2line --exe
ObjectFile`` or ``llvm-symbolizer --obj ObjectFile`` to get file names and line
numbers:

.. code-block:: console

    % sancov.py print a.out.22679.sancov a.out.22673.sancov 2> /dev/null | llvm-symbolizer --obj a.out
    cov.cc:3
    cov.cc:5

Sancov Tool
===========

A new experimental ``sancov`` tool is developed to process coverage files.
The tool is part of LLVM project and is currently supported only on Linux.
It can handle symbolization tasks autonomously without any extra support
from the environment. You need to pass .sancov files (named 
``<module_name>.<pid>.sancov`` and paths to all corresponding binary elf files. 
Sancov matches these files using module names and binaries file names.

.. code-block:: console

    USAGE: sancov [options] <action> (<binary file>|<.sancov file>)...

    Action (required)
      -print                    - Print coverage addresses
      -covered-functions        - Print all covered functions.
      -not-covered-functions    - Print all not covered functions.
      -html-report              - Print HTML coverage report.

    Options
      -blacklist=<string>         - Blacklist file (sanitizer blacklist format).
      -demangle                   - Print demangled function name.
      -strip_path_prefix=<string> - Strip this prefix from file paths in reports


Automatic HTML Report Generation
================================

If ``*SAN_OPTIONS`` contains ``html_cov_report=1`` option set, then html
coverage report would be automatically generated alongside the coverage files.
The ``sancov`` binary should be present in ``PATH`` or
``sancov_path=<path_to_sancov`` option can be used to specify tool location.


How good is the coverage?
=========================

It is possible to find out which PCs are not covered, by subtracting the covered
set from the set of all instrumented PCs. The latter can be obtained by listing
all callsites of ``__sanitizer_cov()`` in the binary. On Linux, ``sancov.py``
can do this for you. Just supply the path to binary and a list of covered PCs:

.. code-block:: console

    % sancov.py print a.out.12345.sancov > covered.txt
    sancov.py: read 2 64-bit PCs from a.out.12345.sancov
    sancov.py: 1 file merged; 2 PCs total
    % sancov.py missing a.out < covered.txt
    sancov.py: found 3 instrumented PCs in a.out
    sancov.py: read 2 PCs from stdin
    sancov.py: 1 PCs missing from coverage
    0x4cc61c

Edge coverage
=============

Consider this code:

.. code-block:: c++

    void foo(int *a) {
      if (a)
        *a = 0;
    }

It contains 3 basic blocks, let's name them A, B, C:

.. code-block:: none

    A
    |\
    | \
    |  B
    | /
    |/
    C

If blocks A, B, and C are all covered we know for certain that the edges A=>B
and B=>C were executed, but we still don't know if the edge A=>C was executed.
Such edges of control flow graph are called
`critical <http://en.wikipedia.org/wiki/Control_flow_graph#Special_edges>`_. The
edge-level coverage (``-fsanitize-coverage=edge``) simply splits all critical
edges by introducing new dummy blocks and then instruments those blocks:

.. code-block:: none

    A
    |\
    | \
    D  B
    | /
    |/
    C

Bitset
======

When ``coverage_bitset=1`` run-time flag is given, the coverage will also be
dumped as a bitset (text file with 1 for blocks that have been executed and 0
for blocks that were not).

.. code-block:: console

    % clang++ -fsanitize=address -fsanitize-coverage=edge cov.cc
    % ASAN_OPTIONS="coverage=1:coverage_bitset=1" ./a.out
    main
    % ASAN_OPTIONS="coverage=1:coverage_bitset=1" ./a.out 1
    foo
    main
    % head *bitset*
    ==> a.out.38214.bitset-sancov <==
    01101
    ==> a.out.6128.bitset-sancov <==
    11011%

For a given executable the length of the bitset is always the same (well,
unless dlopen/dlclose come into play), so the bitset coverage can be
easily used for bitset-based corpus distillation.

Caller-callee coverage
======================

(Experimental!)
Every indirect function call is instrumented with a run-time function call that
captures caller and callee.  At the shutdown time the process dumps a separate
file called ``caller-callee.PID.sancov`` which contains caller/callee pairs as
pairs of lines (odd lines are callers, even lines are callees)

.. code-block:: console

    a.out 0x4a2e0c
    a.out 0x4a6510
    a.out 0x4a2e0c
    a.out 0x4a87f0

Current limitations:

* Only the first 14 callees for every caller are recorded, the rest are silently
  ignored.
* The output format is not very compact since caller and callee may reside in
  different modules and we need to spell out the module names.
* The routine that dumps the output is not optimized for speed
* Only Linux x86_64 is tested so far.
* Sandboxes are not supported.

Coverage counters
=================

This experimental feature is inspired by
`AFL <http://lcamtuf.coredump.cx/afl/technical_details.txt>`__'s coverage
instrumentation. With additional compile-time and run-time flags you can get
more sensitive coverage information.  In addition to boolean values assigned to
every basic block (edge) the instrumentation will collect imprecise counters.
On exit, every counter will be mapped to a 8-bit bitset representing counter
ranges: ``1, 2, 3, 4-7, 8-15, 16-31, 32-127, 128+`` and those 8-bit bitsets will
be dumped to disk.

.. code-block:: console

    % clang++ -g cov.cc -fsanitize=address -fsanitize-coverage=edge,8bit-counters
    % ASAN_OPTIONS="coverage=1:coverage_counters=1" ./a.out
    % ls -l *counters-sancov
    ... a.out.17110.counters-sancov
    % xxd *counters-sancov
    0000000: 0001 0100 01

These counters may also be used for in-process coverage-guided fuzzers. See
``include/sanitizer/coverage_interface.h``:

.. code-block:: c++

    // The coverage instrumentation may optionally provide imprecise counters.
    // Rather than exposing the counter values to the user we instead map
    // the counters to a bitset.
    // Every counter is associated with 8 bits in the bitset.
    // We define 8 value ranges: 1, 2, 3, 4-7, 8-15, 16-31, 32-127, 128+
    // The i-th bit is set to 1 if the counter value is in the i-th range.
    // This counter-based coverage implementation is *not* thread-safe.

    // Returns the number of registered coverage counters.
    uintptr_t __sanitizer_get_number_of_counters();
    // Updates the counter 'bitset', clears the counters and returns the number of
    // new bits in 'bitset'.
    // If 'bitset' is nullptr, only clears the counters.
    // Otherwise 'bitset' should be at least
    // __sanitizer_get_number_of_counters bytes long and 8-aligned.
    uintptr_t
    __sanitizer_update_counter_bitset_and_clear_counters(uint8_t *bitset);

Tracing basic blocks
====================
Experimental support for basic block (or edge) tracing.
With ``-fsanitize-coverage=trace-bb`` the compiler will insert
``__sanitizer_cov_trace_basic_block(s32 *id)`` before every function, basic block, or edge
(depending on the value of ``-fsanitize-coverage=[func,bb,edge]``).
Example:

.. code-block:: console

    % clang -g -fsanitize=address -fsanitize-coverage=edge,trace-bb foo.cc
    % ASAN_OPTIONS=coverage=1 ./a.out

This will produce two files after the process exit:
`trace-points.PID.sancov` and `trace-events.PID.sancov`.
The first file will contain a textual description of all the instrumented points in the program
in the form that you can feed into llvm-symbolizer (e.g. `a.out 0x4dca89`), one per line.
The second file will contain the actual execution trace as a sequence of 4-byte integers
-- these integers are the indices into the array of instrumented points (the first file).

Basic block tracing is currently supported only for single-threaded applications.


Tracing PCs
===========
*Experimental* feature similar to tracing basic blocks, but with a different API.
With ``-fsanitize-coverage=trace-pc`` the compiler will insert
``__sanitizer_cov_trace_pc()`` on every edge.
With an additional ``...=trace-pc,indirect-calls`` flag
``__sanitizer_cov_trace_pc_indirect(void *callee)`` will be inserted on every indirect call.
These callbacks are not implemented in the Sanitizer run-time and should be defined
by the user. So, these flags do not require the other sanitizer to be used.
This mechanism is used for fuzzing the Linux kernel (https://github.com/google/syzkaller)
and can be used with `AFL <http://lcamtuf.coredump.cx/afl>`__.

Tracing PCs with guards
=======================
Another *experimental* feature that tries to combine the functionality of `trace-pc`,
`8bit-counters` and boolean coverage.

With ``-fsanitize-coverage=trace-pc-guard`` the compiler will insert the following code
on every edge:

.. code-block:: none

   if (guard_variable >= 0)
     __sanitizer_cov_trace_pc_guard(&guard_variable)

Every edge will have its own 8-byte `guard_variable`.

The compler will also insert a module constructor that will call

.. code-block:: c++

   // The guards are [start, stop).
   // This function may be called multiple times with the same values of start/stop.
   __sanitizer_cov_trace_pc_guard_init(uint64_t *start, uint64_t *stop);

Similarly to `trace-pc,indirect-calls`, with `trace-pc-guards,indirect-calls`
``__sanitizer_cov_trace_pc_indirect(void *callee)`` will be inserted on every indirect call.

The functions `__sanitizer_cov_trace_pc_*` should be defined by the user.

Tracing data flow
=================

Support for data-flow-guided fuzzing.
With ``-fsanitize-coverage=trace-cmp`` the compiler will insert extra instrumentation
around comparison instructions and switch statements.
Similarly, with ``-fsanitize-coverage=trace-div`` the compiler will instrument
integer division instructions (to capture the right argument of division)
and with  ``-fsanitize-coverage=trace-gep`` --
the `LLVM GEP instructions <http://llvm.org/docs/GetElementPtr.html>`_
(to capture array indices).

.. code-block:: c++

  // Called before a comparison instruction.
  // Arg1 and Arg2 are arguments of the comparison.
  void __sanitizer_cov_trace_cmp1(uint8_t Arg1, uint8_t Arg2);
  void __sanitizer_cov_trace_cmp2(uint16_t Arg1, uint16_t Arg2);
  void __sanitizer_cov_trace_cmp4(uint32_t Arg1, uint32_t Arg2);
  void __sanitizer_cov_trace_cmp8(uint64_t Arg1, uint64_t Arg2);

  // Called before a switch statement.
  // Val is the switch operand.
  // Cases[0] is the number of case constants.
  // Cases[1] is the size of Val in bits.
  // Cases[2:] are the case constants.
  void __sanitizer_cov_trace_switch(uint64_t Val, uint64_t *Cases);

  // Called before a division statement.
  // Val is the second argument of division.
  void __sanitizer_cov_trace_div4(uint32_t Val);
  void __sanitizer_cov_trace_div8(uint64_t Val);

  // Called before a GetElemementPtr (GEP) instruction
  // for every non-constant array index.
  void __sanitizer_cov_trace_gep(uintptr_t Idx);


This interface is a subject to change.
The current implementation is not thread-safe and thus can be safely used only for single-threaded targets.

Output directory
================

By default, .sancov files are created in the current working directory.
This can be changed with ``ASAN_OPTIONS=coverage_dir=/path``:

.. code-block:: console

    % ASAN_OPTIONS="coverage=1:coverage_dir=/tmp/cov" ./a.out foo
    % ls -l /tmp/cov/*sancov
    -rw-r----- 1 kcc eng 4 Nov 27 12:21 a.out.22673.sancov
    -rw-r----- 1 kcc eng 8 Nov 27 12:21 a.out.22679.sancov

Sudden death
============

Normally, coverage data is collected in memory and saved to disk when the
program exits (with an ``atexit()`` handler), when a SIGSEGV is caught, or when
``__sanitizer_cov_dump()`` is called.

If the program ends with a signal that ASan does not handle (or can not handle
at all, like SIGKILL), coverage data will be lost. This is a big problem on
Android, where SIGKILL is a normal way of evicting applications from memory.

With ``ASAN_OPTIONS=coverage=1:coverage_direct=1`` coverage data is written to a
memory-mapped file as soon as it collected.

.. code-block:: console

    % ASAN_OPTIONS="coverage=1:coverage_direct=1" ./a.out
    main
    % ls
    7036.sancov.map  7036.sancov.raw  a.out
    % sancov.py rawunpack 7036.sancov.raw
    sancov.py: reading map 7036.sancov.map
    sancov.py: unpacking 7036.sancov.raw
    writing 1 PCs to a.out.7036.sancov
    % sancov.py print a.out.7036.sancov
    sancov.py: read 1 PCs from a.out.7036.sancov
    sancov.py: 1 files merged; 1 PCs total
    0x4b2bae

Note that on 64-bit platforms, this method writes 2x more data than the default,
because it stores full PC values instead of 32-bit offsets.

In-process fuzzing
==================

Coverage data could be useful for fuzzers and sometimes it is preferable to run
a fuzzer in the same process as the code being fuzzed (in-process fuzzer).

You can use ``__sanitizer_get_total_unique_coverage()`` from
``<sanitizer/coverage_interface.h>`` which returns the number of currently
covered entities in the program. This will tell the fuzzer if the coverage has
increased after testing every new input.

If a fuzzer finds a bug in the ASan run, you will need to save the reproducer
before exiting the process.  Use ``__asan_set_death_callback`` from
``<sanitizer/asan_interface.h>`` to do that.

An example of such fuzzer can be found in `the LLVM tree
<http://llvm.org/viewvc/llvm-project/llvm/trunk/lib/Fuzzer/README.txt?view=markup>`_.

Performance
===========

This coverage implementation is **fast**. With function-level coverage
(``-fsanitize-coverage=func``) the overhead is not measurable. With
basic-block-level coverage (``-fsanitize-coverage=bb``) the overhead varies
between 0 and 25%.

==============  =========  =========  =========  =========  =========  =========
     benchmark      cov0        cov1   diff 0-1       cov2   diff 0-2   diff 1-2
==============  =========  =========  =========  =========  =========  =========
 400.perlbench    1296.00    1307.00       1.01    1465.00       1.13       1.12
     401.bzip2     858.00     854.00       1.00    1010.00       1.18       1.18
       403.gcc     613.00     617.00       1.01     683.00       1.11       1.11
       429.mcf     605.00     582.00       0.96     610.00       1.01       1.05
     445.gobmk     896.00     880.00       0.98    1050.00       1.17       1.19
     456.hmmer     892.00     892.00       1.00     918.00       1.03       1.03
     458.sjeng     995.00    1009.00       1.01    1217.00       1.22       1.21
462.libquantum     497.00     492.00       0.99     534.00       1.07       1.09
   464.h264ref    1461.00    1467.00       1.00    1543.00       1.06       1.05
   471.omnetpp     575.00     590.00       1.03     660.00       1.15       1.12
     473.astar     658.00     652.00       0.99     715.00       1.09       1.10
 483.xalancbmk     471.00     491.00       1.04     582.00       1.24       1.19
      433.milc     616.00     627.00       1.02     627.00       1.02       1.00
      444.namd     602.00     601.00       1.00     654.00       1.09       1.09
    447.dealII     630.00     634.00       1.01     653.00       1.04       1.03
    450.soplex     365.00     368.00       1.01     395.00       1.08       1.07
    453.povray     427.00     434.00       1.02     495.00       1.16       1.14
       470.lbm     357.00     375.00       1.05     370.00       1.04       0.99
   482.sphinx3     927.00     928.00       1.00    1000.00       1.08       1.08
==============  =========  =========  =========  =========  =========  =========

Why another coverage?
=====================

Why did we implement yet another code coverage?
  * We needed something that is lightning fast, plays well with
    AddressSanitizer, and does not significantly increase the binary size.
  * Traditional coverage implementations based in global counters
    `suffer from contention on counters
    <https://groups.google.com/forum/#!topic/llvm-dev/cDqYgnxNEhY>`_.
