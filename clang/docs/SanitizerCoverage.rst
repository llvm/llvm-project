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

At run time, pass ``coverage=1`` in ``ASAN_OPTIONS``,
``LSAN_OPTIONS``, ``MSAN_OPTIONS`` or ``UBSAN_OPTIONS``, as
appropriate. For the standalone coverage mode, use ``UBSAN_OPTIONS``.

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
      -symbolize                - Symbolizes the report.

    Options
      -blacklist=<string>         - Blacklist file (sanitizer blacklist format).
      -demangle                   - Print demangled function name.
      -strip_path_prefix=<string> - Strip this prefix from file paths in reports


Coverage Reports (Experimental)
================================

``.sancov`` files do not contain enough information to generate a source-level 
coverage report. The missing information is contained
in debug info of the binary. Thus the ``.sancov`` has to be symbolized
to produce a ``.symcov`` file first:

.. code-block:: console

    sancov -symbolize my_program.123.sancov my_program > my_program.123.symcov

The ``.symcov`` file can be browsed overlayed over the source code by
running ``tools/sancov/coverage-report-server.py`` script that will start
an HTTP server.


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

With ``-fsanitize-coverage=trace-pc-guard`` the compiler will insert the following code
on every edge:

.. code-block:: none

   __sanitizer_cov_trace_pc_guard(&guard_variable)

Every edge will have its own `guard_variable` (uint32_t).

The compler will also insert a module constructor that will call

.. code-block:: c++

   // The guards are [start, stop).
   // This function will be called at least once per DSO and may be called
   // more than once with the same values of start/stop.
   __sanitizer_cov_trace_pc_guard_init(uint32_t *start, uint32_t *stop);

With `trace-pc-guards,indirect-calls`
``__sanitizer_cov_trace_pc_indirect(void *callee)`` will be inserted on every indirect call.

The functions `__sanitizer_cov_trace_pc_*` should be defined by the user.

Example: 

.. code-block:: c++

  // trace-pc-guard-cb.cc
  #include <stdint.h>
  #include <stdio.h>
  #include <sanitizer/coverage_interface.h>

  // This callback is inserted by the compiler as a module constructor
  // into every DSO. 'start' and 'stop' correspond to the
  // beginning and end of the section with the guards for the entire
  // binary (executable or DSO). The callback will be called at least
  // once per DSO and may be called multiple times with the same parameters.
  extern "C" void __sanitizer_cov_trace_pc_guard_init(uint32_t *start,
                                                      uint32_t *stop) {
    static uint64_t N;  // Counter for the guards.
    if (start == stop || *start) return;  // Initialize only once.
    printf("INIT: %p %p\n", start, stop);
    for (uint32_t *x = start; x < stop; x++)
      *x = ++N;  // Guards should start from 1.
  }

  // This callback is inserted by the compiler on every edge in the
  // control flow (some optimizations apply).
  // Typically, the compiler will emit the code like this:
  //    if(*guard)
  //      __sanitizer_cov_trace_pc_guard(guard);
  // But for large functions it will emit a simple call:
  //    __sanitizer_cov_trace_pc_guard(guard);
  extern "C" void __sanitizer_cov_trace_pc_guard(uint32_t *guard) {
    if (!*guard) return;  // Duplicate the guard check.
    // If you set *guard to 0 this code will not be called again for this edge.
    // Now you can get the PC and do whatever you want: 
    //   store it somewhere or symbolize it and print right away.
    // The values of `*guard` are as you set them in
    // __sanitizer_cov_trace_pc_guard_init and so you can make them consecutive
    // and use them to dereference an array or a bit vector.
    void *PC = __builtin_return_address(0);
    char PcDescr[1024];
    // This function is a part of the sanitizer run-time.
    // To use it, link with AddressSanitizer or other sanitizer.
    __sanitizer_symbolize_pc(PC, "%p %F %L", PcDescr, sizeof(PcDescr));
    printf("guard: %p %x PC %s\n", guard, *guard, PcDescr);
  }

.. code-block:: c++

  // trace-pc-guard-example.cc
  void foo() { }
  int main(int argc, char **argv) {
    if (argc > 1) foo();
  }

.. code-block:: console
  
  clang++ -g  -fsanitize-coverage=trace-pc-guard trace-pc-guard-example.cc -c
  clang++ trace-pc-guard-cb.cc trace-pc-guard-example.o -fsanitize=address
  ASAN_OPTIONS=strip_path_prefix=`pwd`/ ./a.out

.. code-block:: console

  INIT: 0x71bcd0 0x71bce0
  guard: 0x71bcd4 2 PC 0x4ecd5b in main trace-pc-guard-example.cc:2
  guard: 0x71bcd8 3 PC 0x4ecd9e in main trace-pc-guard-example.cc:3:7

.. code-block:: console

  ASAN_OPTIONS=strip_path_prefix=`pwd`/ ./a.out with-foo


.. code-block:: console

  INIT: 0x71bcd0 0x71bce0
  guard: 0x71bcd4 2 PC 0x4ecd5b in main trace-pc-guard-example.cc:3
  guard: 0x71bcdc 4 PC 0x4ecdc7 in main trace-pc-guard-example.cc:4:17
  guard: 0x71bcd0 1 PC 0x4ecd20 in foo() trace-pc-guard-example.cc:2:14


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

