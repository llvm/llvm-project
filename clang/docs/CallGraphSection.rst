==================
Call Graph Section
==================

Introduction
============

With ``-fcall-graph-section``, the compiler will create a call graph section 
in the object file. It will include type identifiers for indirect calls and 
targets. This information can be used to map indirect calls to their receivers 
with matching types. A complete and high-precision call graph can be 
reconstructed by complementing this information with disassembly 
(see ``llvm-objdump --call-graph-info``).

Semantics
=========

A coarse-grained, type-agnostic call graph may allow indirect calls to target
any function in the program. This approach ensures completeness since no
indirect call edge is missing. However, it is generally poor in precision
due to having unneeded edges.

A call graph section provides type identifiers for indirect calls and targets.
This information can be used to restrict the receivers of an indirect target to
indirect calls with matching type. Consequently, the precision for indirect
call edges are improved while maintaining the completeness.

The ``llvm-objdump`` utility provides a ``--call-graph-info`` option to extract
full call graph information by parsing the content of the call graph section
and disassembling the program for complementary information, e.g., direct
calls.

Section layout
==============

A call graph section consists of zero or more call graph entries.
Each entry contains information on a function and its indirect calls.

An entry of a call graph section has the following layout in the binary:

+---------------------+-----------------------------------------------------------------------+
| Element             | Content                                                               |
+=====================+=======================================================================+
| FormatVersionNumber | Format version number.                                                |
+---------------------+-----------------------------------------------------------------------+
| FunctionEntryPc     | Function entry address.                                               |
+---------------------+-----------------------------------+-----------------------------------+
|                     | A flag whether the function is an | - 0: not an indirect target       |
| FunctionKind        | indirect target, and if so,       | - 1: indirect target, unknown id  |
|                     | whether its type id is known.     | - 2: indirect target, known id    |
+---------------------+-----------------------------------+-----------------------------------+
| FunctionTypeId      | Type id for the indirect target. Present only when FunctionKind is 2. |
+---------------------+-----------------------------------------------------------------------+
| CallSiteCount       | Number of type id to indirect call site mappings that follow.         |
+---------------------+-----------------------------------------------------------------------+
| CallSiteList        | List of type id and indirect call site pc pairs.                      |
+---------------------+-----------------------------------------------------------------------+

Each element in an entry (including each element of the contained lists and
pairs) occupies 64-bit space.

The format version number is repeated per entry to support concatenation of
call graph sections with different format versions by the linker.

As of now, the only supported format version is described above and has version
number 0.

Type identifiers
================

The type for an indirect call or target is the function signature.
The mapping from a type to an identifier is an ABI detail.
In the current experimental implementation, an identifier of type T is
computed as follows:

  -  Obtain the generalized mangled name for “typeinfo name for T”.
  -  Compute MD5 hash of the name as a string.
  -  Reinterpret the first 8 bytes of the hash as a little-endian 64-bit integer.

To avoid mismatched pointer types, generalizations are applied.
Pointers in return and argument types are treated as equivalent as long as the
qualifiers for the type they point to match.
For example, ``char*``, ``char**``, and ``int*`` are considered equivalent
types. However, ``char*`` and ``const char*`` are considered separate types.

Missing type identifiers
========================

For functions, two cases need to be considered. First, if the compiler cannot
deduce a type id for an indirect target, it will be listed as an indirect target
without a type id. Second, if an object without a call graph section gets
linked, the final call graph section will lack information on functions from
the object. For completeness, these functions need to be taken as receiver to
any indirect call regardless of their type id.
``llvm-objdump --call-graph-info`` lists these functions as indirect targets
with `UNKNOWN` type id.

For indirect calls, current implementation guarantees a type id for each
compiled call. However, if an object without a call graph section gets linked,
no type id will be present for its indirect calls. For completeness, these calls
need to be taken to target any indirect target regardless of their type id. For
indirect calls, ``llvm-objdump --call-graph-info`` prints 1) a complete list of
indirect calls, 2) type id to indirect call mappings. The difference of these
lists allow to deduce the indirect calls with missing type ids.

TODO: measure and report the ratio of missed type ids

Performance
===========

A call graph section does not affect the executable code and does not occupy
memory during process execution. Therefore, there is no performance overhead.

The scheme has not yet been optimized for binary size.

TODO: measure and report the increase in the binary size

Example
=======

For example, consider the following C++ code:

.. code-block:: cpp

    namespace {
      // Not an indirect target
      void foo() {}
    }

    // Indirect target 1
    void bar() {}

    // Indirect target 2
    int baz(char a, float *b) {
      return 0;
    }

    // Indirect target 3
    int main() {
      char a;
      float b;
      void (*fp_bar)() = bar;
      int (*fp_baz1)(char, float*) = baz;
      int (*fp_baz2)(char, float*) = baz;

      // Indirect call site 1
      fp_bar();

      // Indirect call site 2
      fp_baz1(a, &b);

      // Indirect call site 3: shares the type id with indirect call site 2
      fp_baz2(a, &b);

      // Direct call sites
      foo();
      bar();
      baz(a, &b);

      return 0;
    }

Following will compile it with a call graph section created in the binary:

.. code-block:: bash

  $ clang -fcall-graph-section example.cpp

During the construction of the call graph section, the type identifiers are 
computed as follows:

+---------------+-----------------------+----------------------------+----------------------------+
| Function name | Generalized signature | Mangled name (itanium ABI) | Numeric type id (md5 hash) |
+===============+=======================+============================+============================+
|  bar          | void ()               | _ZTSFvvE.generalized       | f85c699bb8ef20a2           |
+---------------+-----------------------+----------------------------+----------------------------+
|  baz          | int (char, void*)     | _ZTSFicPvE.generalized     | e3804d2a7f2b03fe           |
+---------------+-----------------------+----------------------------+----------------------------+
|  main         | int ()                | _ZTSFivE.generalized       | a9494def81a01dc            |
+---------------+-----------------------+----------------------------+----------------------------+

The call graph section will have the following content:

+---------------+-----------------+--------------+----------------+---------------+--------------------------------------+
| FormatVersion | FunctionEntryPc | FunctionKind | FunctionTypeId | CallSiteCount | CallSiteList                         |
+===============+=================+==============+================+===============+======================================+
| 0             | EntryPc(foo)    | 0            | (empty)        | 0             | (empty)                              |
+---------------+-----------------+--------------+----------------+---------------+--------------------------------------+
| 0             | EntryPc(bar)    | 2            | TypeId(bar)    | 0             | (empty)                              |
+---------------+-----------------+--------------+----------------+---------------+--------------------------------------+
| 0             | EntryPc(baz)    | 2            | TypeId(baz)    | 0             | (empty)                              |
+---------------+-----------------+--------------+----------------+---------------+--------------------------------------+
| 0             | EntryPc(main)   | 2            | TypeId(main)   | 3             | * TypeId(bar), CallSitePc(fp_bar())  |
|               |                 |              |                |               | * TypeId(baz), CallSitePc(fp_baz1()) |
|               |                 |              |                |               | * TypeId(baz), CallSitePc(fp_baz2()) |
+---------------+-----------------+--------------+----------------+---------------+--------------------------------------+


The ``llvm-objdump`` utility can parse the call graph section and disassemble
the program to provide complete call graph information. This includes any
additional call sites from the binary:

.. code-block:: bash

    $ llvm-objdump --call-graph-info a.out

    # Comments are not a part of the llvm-objdump's output but inserted for clarifications.

    a.out:  file format elf64-x86-64
    # These warnings are due to the functions and the indirect calls coming from linked objects.
    llvm-objdump: warning: 'a.out': callgraph section does not have type ids for 3 indirect calls
    llvm-objdump: warning: 'a.out': callgraph section does not have information for 10 functions

    # Unknown targets are the 10 functions the warnings mention.
    INDIRECT TARGET TYPES (TYPEID [FUNC_ADDR,])
    UNKNOWN 401000 401100 401234 401050 401090 4010d0 4011d0 401020 401060 401230
    a9494def81a01dc 401150            # main()
    f85c699bb8ef20a2 401120           # bar()
    e3804d2a7f2b03fe 401130           # baz()

    # Notice that the call sites share the same type id as target functions
    INDIRECT CALL TYPES (TYPEID [CALL_SITE_ADDR,])
    f85c699bb8ef20a2 401181           # Indirect call site 1 (fp_bar())
    e3804d2a7f2b03fe 401191 4011a1    # Indirect call site 2 and 3 (fp_baz1() and fp_baz2())

    INDIRECT CALL SITES (CALLER_ADDR [CALL_SITE_ADDR,])
    401000 401012                     # _init
    401150 401181 401191 4011a1       # main calls fp_bar(), fp_baz1(), fp_baz2()
    4011d0 401215                     # __libc_csu_init
    401020 40104a                     # _start

    DIRECT CALL SITES (CALLER_ADDR [(CALL_SITE_ADDR, TARGET_ADDR),])
    4010d0 4010e2 401060              # __do_global_dtors_aux
    401150 4011a6 401110 4011ab 401120 4011ba 401130   # main calls foo(), bar(), baz()
    4011d0 4011fd 401000              # __libc_csu_init

    FUNCTIONS (FUNC_ENTRY_ADDR, SYM_NAME)
    401000 _init
    401100 frame_dummy
    401234 _fini
    401050 _dl_relocate_static_pie
    401090 register_tm_clones
    4010d0 __do_global_dtors_aux
    401110 _ZN12_GLOBAL__N_13fooEv    # (anonymous namespace)::foo()
    401150 main                       # main
    4011d0 __libc_csu_init
    401020 _start
    401060 deregister_tm_clones
    401120 _Z3barv                    # bar()
    401130 _Z3bazcPf                  # baz(char, float*)
    401230 __libc_csu_fini
