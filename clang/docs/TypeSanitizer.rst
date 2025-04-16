=============
TypeSanitizer
=============

.. contents::
   :local:

Introduction
============

The TypeSanitizer is a detector for strict type aliasing violations. It consists of a compiler
instrumentation module and a run-time library. C/C++ has type-based aliasing rules, and LLVM 
can exploit these for optimizations given the TBAA metadata Clang emits. In general, a pointer 
of a given type cannot access an object of a different type, with only a few exceptions. 

These rules aren't always apparent to users, which leads to code that violates these rules
(e.g. for type punning). This can lead to optimization passes introducing bugs unless the 
code is build with ``-fno-strict-aliasing``, sacrificing performance.

TypeSanitizer is built to catch when these strict aliasing rules have been violated, helping 
users find where such bugs originate in their code despite the code looking valid at first glance.

As TypeSanitizer is still experimental, it can currently have a large impact on runtime speed, 
memory use, and code size. It also has a large compile-time overhead. Work is being done to 
reduce these impacts.

The TypeSanitizer Algorithm
===========================
For each TBAA type-access descriptor, encoded in LLVM IR using TBAA Metadata, the instrumentation 
pass generates descriptor tales. Thus there is a unique pointer to each type (and access descriptor).
These tables are comdat (except for anonymous-namespace types), so the pointer values are unique 
across the program.

The descriptors refer to other descriptors to form a type aliasing tree, like how LLVM's TBAA data 
does.

The runtime uses 8 bytes of shadow memory, the size of the pointer to the type descriptor, for 
every byte of accessed data in the program. The first byte of a type will have its shadow memory 
be set to the pointer to its type descriptor. Aside from that, there are some other values it may be.

* 0 is used to represent an unknown type
* Negative numbers represent an interior byte: A byte inside a type that is not the first one. As an 
  example, a value of -2 means you are in the third byte of a type.

The Instrumentation first checks for an exact match between the type of the current access and the 
type for that address in the shadow memory. This can quickly be done by checking pointer values. If 
it matches, it checks the remaining shadow memory of the type to ensure they are the correct negative 
numbers. If this fails, it calls the "slow path" check. If the exact match fails, we check to see if 
the value, and the remainder of the shadow bytes, is 0. If they are, we can set the shadow memory to 
the correct type descriptor pointer for the first byte, and the correct negative numbers for the rest 
of the type's shadow.

If the type in shadow memory is neither an exact match nor 0, we call the slower runtime check. It 
uses the full TBAA algorithm, just as the compiler does, to determine when two types are permitted to 
alias.

The instrumentation pass inserts calls to the memset intrinsic to set the memory updated by memset, 
memcpy, and memmove, as well as allocas/byval (and for lifetime.start/end) to reset the shadow memory 
to reflect that the type is now unknown. The runtime intercepts memset, memcpy, etc. to perform the 
same function for the library calls.

How to build
============

Build LLVM/Clang with `CMake <https://llvm.org/docs/CMake.html>`_ and enable
the ``compiler-rt`` runtime. An example CMake configuration that will allow
for the use/testing of TypeSanitizer:

.. code-block:: console

   $ cmake -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang" -DLLVM_ENABLE_RUNTIMES="compiler-rt" <path to source>/llvm

Usage
=====

Compile and link your program with ``-fsanitize=type`` flag. The
TypeSanitizer run-time library should be linked to the final executable, so
make sure to use ``clang`` (not ``ld``) for the final link step. To
get a reasonable performance add ``-O1`` or higher.
TypeSanitizer by default doesn't print the full stack trace in error messages. Use ``TYSAN_OPTIONS=print_stacktrace=1`` 
to print the full trace. To get nicer stack traces in error messages add ``-fno-omit-frame-pointer`` and 
``-g``.  To get perfect stack traces you may need to disable inlining (just use ``-O1``) and tail call elimination 
(``-fno-optimize-sibling-calls``).

.. code-block:: console

    % cat example_AliasViolation.c
    int main(int argc, char **argv) {
      int x = 100;
      float *y = (float*)&x;
      *y += 2.0f;          // Strict aliasing violation
      return 0;
    }

    # Compile and link
    % clang++ -g -fsanitize=type example_AliasViolation.cc

The program will print an error message to ``stderr`` each time a strict aliasing violation is detected. 
The program won't terminate, which will allow you to detect many strict aliasing violations in one 
run.

.. code-block:: console

    % ./a.out
    ==1375532==ERROR: TypeSanitizer: type-aliasing-violation on address 0x7ffeebf1a72c (pc 0x5b3b1145ff41 bp 0x7ffeebf1a660 sp 0x7ffeebf19e08 tid 1375532)
    READ of size 4 at 0x7ffeebf1a72c with type float accesses an existing object of type int
        #0 0x5b3b1145ff40 in main example_AliasViolation.c:4:10

    ==1375532==ERROR: TypeSanitizer: type-aliasing-violation on address 0x7ffeebf1a72c (pc 0x5b3b1146008a bp 0x7ffeebf1a660 sp 0x7ffeebf19e08 tid 1375532)
    WRITE of size 4 at 0x7ffeebf1a72c with type float accesses an existing object of type int
        #0 0x5b3b11460089 in main example_AliasViolation.c:4:10

Error terminology
------------------

There are some terms that may appear in TypeSanitizer errors that are derived from 
`TBAA Metadata <https://llvm.org/docs/LangRef.html#tbaa-metadata>`. This section hopes to provide a 
brief dictionary of these terms.

* ``omnipotent char``: This is a special type which can alias with anything. Its name comes from the C/C++ 
  type ``char``.
* ``type p[x]``: This signifies pointers to the type. ``x`` is the number of indirections to reach the final value.
  As an example, a pointer to a pointer to an integer would be ``type p2 int``.

TypeSanitizer is still experimental. User-facing error messages should be improved in the future to remove 
references to LLVM IR specific terms.

Sanitizer features
==================

``__has_feature(type_sanitizer)``
------------------------------------

In some cases one may need to execute different code depending on whether
TypeSanitizer is enabled.
:ref:`\_\_has\_feature <langext-__has_feature-__has_extension>` can be used for
this purpose.

.. code-block:: c

    #if defined(__has_feature)
    #  if __has_feature(type_sanitizer)
    // code that builds only under TypeSanitizer
    #  endif
    #endif

``__attribute__((no_sanitize("type")))``
-----------------------------------------------

Some code you may not want to be instrumented by TypeSanitizer.  One may use the
function attribute ``no_sanitize("type")`` to disable instrumenting type aliasing. 
It is possible, depending on what happens in non-instrumented code, that instrumented code 
emits false-positives/ false-negatives. This attribute may not be supported by other 
compilers, so we suggest to use it together with ``__has_feature(type_sanitizer)``.

``__attribute__((disable_sanitizer_instrumentation))``
--------------------------------------------------------

The ``disable_sanitizer_instrumentation`` attribute can be applied to functions
to prevent all kinds of instrumentation. As a result, it may introduce false
positives and incorrect stack traces. Therefore, it should be used with care,
and only if absolutely required; for example for certain code that cannot
tolerate any instrumentation and resulting side-effects. This attribute
overrides ``no_sanitize("type")``.

Ignorelist
----------

TypeSanitizer supports ``src`` and ``fun`` entity types in
:doc:`SanitizerSpecialCaseList`, that can be used to suppress aliasing 
violation reports in the specified source files or functions. Like 
with other methods of ignoring instrumentation, this can result in false 
positives/ false-negatives.

Limitations
-----------

* TypeSanitizer uses more real memory than a native run. It uses 8 bytes of
  shadow memory for each byte of user memory.
* There are transformation passes which run before TypeSanitizer. If these 
  passes optimize out an aliasing violation, TypeSanitizer cannot catch it.
* Currently, all instrumentation is inlined. This can result in a **15x** 
  (on average) increase in generated file size, and **3x** to **7x** increase 
  in compile time. In some documented cases this can cause the compiler to hang.
  There are plans to improve this in the future.
* Codebases that use unions and struct-initialized variables can see incorrect 
  results, as TypeSanitizer doesn't yet instrument these reliably.
* Since Clang & LLVM's TBAA system is used to generate the checks used by the 
  instrumentation, TypeSanitizer follows Clang & LLVM's rules for type aliasing. 
  There may be situations where that disagrees with the standard. However this 
  does at least mean that TypeSanitizer will catch any aliasing violations that  
  would cause bugs when compiling with Clang & LLVM.
* TypeSanitizer cannot currently be run alongside other sanitizers such as 
  AddressSanitizer, ThreadSanitizer or UndefinedBehaviourSanitizer.

Current Status
--------------

TypeSanitizer is brand new, and still in development. There are some known 
issues, especially in areas where Clang's emitted TBAA data isn't extensive 
enough for TypeSanitizer's runtime.

We are actively working on enhancing the tool --- stay tuned.  Any help, 
issues, pull requests, ideas, is more than welcome. You can find the 
`issue tracker here. <https://github.com/llvm/llvm-project/issues?q=is%3Aissue%20state%3Aopen%20TySan%20label%3Acompiler-rt%3Atysan>`_
