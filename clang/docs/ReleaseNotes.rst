=========================
Clang 8.0.0 Release Notes
=========================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <https://llvm.org/>`_

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C/OpenCL
frontend, part of the LLVM Compiler Infrastructure, release 8.0.0. Here we
describe the status of Clang in some detail, including major
improvements from the previous release and new feature work. For the
general LLVM release notes, see `the LLVM
documentation <https://llvm.org/docs/ReleaseNotes.html>`_. All LLVM
releases may be downloaded
from the `LLVM releases web site <https://releases.llvm.org/>`_.

For more information about Clang or LLVM, including information about the
latest release, please see the `Clang Web Site <https://clang.llvm.org>`_ or the
`LLVM Web Site <https://llvm.org>`_.

What's New in Clang 8.0.0?
==========================

Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

Major New Features
------------------

- Clang supports use of a profile remapping file, which permits
  profile data captured for one version of a program to be applied
  when building another version where symbols have changed (for
  example, due to renaming a class or namespace).
  See the :ref:`UsersManual <profile_remapping>` for details.

- Clang has new options to initialize automatic variables with a pattern. The default is still that automatic variables are uninitialized. This isn't meant to change the semantics of C and C++. Rather, it's meant to be a last resort when programmers inadvertently have some undefined behavior in their code. These options aim to make undefined behavior hurt less, which security-minded people will be very happy about. Notably, this means that there's no inadvertent information leak when:

    * The compiler re-uses stack slots, and a value is used uninitialized.

    * The compiler re-uses a register, and a value is used uninitialized.

    * Stack structs / arrays / unions with padding are copied.

  These options only address stack and register information leaks.

  Caveats:

    * Variables declared in unreachable code and used later aren't initialized. This affects goto statements, Duff's device, and other objectionable uses of switch statements. This should instead be a hard-error in any serious codebase.

    * These options don't affect volatile stack variables.

    * Padding isn't fully handled yet.

  How to use it on the command line:

    * ``-ftrivial-auto-var-init=uninitialized`` (the default)

    * ``-ftrivial-auto-var-init=pattern``

  There is also a new attribute to request a variable to not be initialized, mainly to disable initialization of large stack arrays when deemed too expensive:

    * ``int dont_initialize_me __attribute((uninitialized));``


Improvements to Clang's diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``-Wextra-semi-stmt`` is a new diagnostic that diagnoses extra semicolons,
  much like ``-Wextra-semi``. This new diagnostic diagnoses all *unnecessary*
  null statements (expression statements without an expression), unless: the
  semicolon directly follows a macro that was expanded to nothing or if the
  semicolon is within the macro itself. This applies to macros defined in system
  headers as well as user-defined macros.

  .. code-block:: c++

      #define MACRO(x) int x;
      #define NULLMACRO(varname)

      void test() {
        ; // <- warning: ';' with no preceding expression is a null statement

        while (true)
          ; // OK, it is needed.

        switch (my_enum) {
        case E1:
          // stuff
          break;
        case E2:
          ; // OK, it is needed.
        }

        MACRO(v0;) // Extra semicolon, but within macro, so ignored.

        MACRO(v1); // <- warning: ';' with no preceding expression is a null statement

        NULLMACRO(v2); // ignored, NULLMACRO expanded to nothing.
      }

- ``-Wempty-init-stmt`` is a new diagnostic that diagnoses empty init-statements
  of ``if``, ``switch``, ``range-based for``, unless: the semicolon directly
  follows a macro that was expanded to nothing or if the semicolon is within the
  macro itself (both macros from system headers, and normal macros). This
  diagnostic is in the ``-Wextra-semi-stmt`` group and is enabled in
  ``-Wextra``.

  .. code-block:: c++

      void test() {
        if(; // <- warning: init-statement of 'if' is a null statement
           true)
          ;

        switch (; // <- warning: init-statement of 'switch' is a null statement
                x) {
          ...
        }

        for (; // <- warning: init-statement of 'range-based for' is a null statement
             int y : S())
          ;
      }


Non-comprehensive list of changes in this release
-------------------------------------------------

- The experimental feature Pretokenized Headers (PTH) was removed in its
  entirely from Clang. The feature did not properly work with about 1/3 of the
  possible tokens available and was unmaintained.

- The internals of libc++ include directory detection on MacOS have changed.
  Instead of running a search based on the ``-resource-dir`` flag, the search
  is now based on the path of the compiler in the filesystem. The default
  behaviour should not change. However, if you override ``-resource-dir``
  manually and rely on the old behaviour you will need to add appropriate
  compiler flags for finding the corresponding libc++ include directory.

- The integrated assembler is used now by default for all MIPS targets.

- Improved support for MIPS N32 ABI and MIPS R6 target triples.

- Clang now includes builtin functions for bitwise rotation of common value
  sizes, such as: `__builtin_rotateleft32
  <LanguageExtensions.html#builtin-rotateleft>`_

- Improved optimization for the corresponding MSVC compatibility builtins such
  as ``_rotl()``.

New Compiler Flags
------------------

- ``-mspeculative-load-hardening`` Clang now has an option to enable
  Speculative Load Hardening.

- ``-fprofile-filter-files=[regexes]`` and ``-fprofile-exclude-files=[regexes]``.

  Clang has now options to filter or exclude some files when
  instrumenting for gcov-based profiling.
  See the `UsersManual <UsersManual.html#cmdoption-fprofile-filter-files>`_ for details.

- When using a custom stack alignment, the ``stackrealign`` attribute is now
  implicitly set on the main function.

- Emission of ``R_MIPS_JALR`` and ``R_MICROMIPS_JALR`` relocations can now
  be controlled by the ``-mrelax-pic-calls`` and ``-mno-relax-pic-calls``
  options.

Modified Compiler Flags
-----------------------

- As of clang 8, ``alignof`` and ``_Alignof`` return the ABI alignment of a type,
  as opposed to the preferred alignment. ``__alignof`` still returns the
  preferred alignment. ``-fclang-abi-compat=7`` (and previous) will make
  ``alignof`` and ``_Alignof`` return preferred alignment again.


New Pragmas in Clang
--------------------

- Clang now supports adding multiple `#pragma clang attribute` attributes into
  a scope of pushed attributes.

Attribute Changes in Clang
--------------------------

* Clang now supports enabling/disabling speculative load hardening on a
  per-function basis using the function attribute
  ``speculative_load_hardening``/``no_speculative_load_hardening``.

Windows Support
---------------

- clang-cl now supports the use of the precompiled header options ``/Yc`` and ``/Yu``
  without the filename argument. When these options are used without the
  filename, a `#pragma hdrstop` inside the source marks the end of the
  precompiled code.

- clang-cl has a new command-line option, ``/Zc:dllexportInlines-``, similar to
  ``-fvisibility-inlines-hidden`` on non-Windows, that makes class-level
  `dllexport` and `dllimport` attributes not apply to inline member functions.
  This can significantly reduce compile and link times. See the `User's Manual
  <UsersManual.html#the-zc-dllexportinlines-option>`_ for more info.

- For MinGW, ``-municode`` now correctly defines ``UNICODE`` during
  preprocessing.

- For MinGW, clang now produces vtables and RTTI for dllexported classes
  without key functions. This fixes building Qt in debug mode.

- Allow using Address Sanitizer and Undefined Behaviour Sanitizer on MinGW.

- Structured Exception Handling support for ARM64 Windows. The ARM64 Windows
  target is in pretty good shape now.


OpenCL Kernel Language Changes in Clang
---------------------------------------

Misc:

- Improved address space support with Clang builtins.

- Improved various diagnostics for vectors with element types from extensions;
  values used in attributes; duplicate address spaces.

- Allow blocks to capture arrays.

- Allow zero assignment and comparisons between variables of ``queue_t`` type.

- Improved diagnostics of formatting specifiers and argument promotions for
  vector types in ``printf``.

- Fixed return type of enqueued kernel and pipe builtins.

- Fixed address space of ``clk_event_t`` generated in the IR.

- Fixed address space when passing/returning structs.

Header file fixes:

- Added missing extension guards around several builtin function overloads.

- Fixed serialization support when registering vendor extensions using pragmas.

- Fixed OpenCL version in declarations of builtin functions with sampler-less
  image accesses.

New vendor extensions added:

- ``cl_intel_planar_yuv``

- ``cl_intel_device_side_avc_motion_estimation``


C++ for OpenCL:

- Added support of address space conversions in C style casts.

- Enabled address spaces for references.

- Fixed use of address spaces in templates: address space deduction and diagnostics.

- Changed default address space to work with C++ specific concepts: class members,
  template parameters, etc.

- Added generic address space by default to the generated hidden 'this' parameter.

- Extend overload ranking rules for address spaces.


ABI Changes in Clang
--------------------

- ``_Alignof`` and ``alignof`` now return the ABI alignment of a type, as opposed
  to the preferred alignment.

  - This is more in keeping with the language of the standards, as well as
    being compatible with gcc
  - ``__alignof`` and ``__alignof__`` still return the preferred alignment of
    a type
  - This shouldn't break any ABI except for things that explicitly ask for
    ``alignas(alignof(T))``.
  - If you have interfaces that break with this change, you may wish to switch
    to ``alignas(__alignof(T))``, instead of using the ``-fclang-abi-compat``
    switch.

OpenMP Support in Clang
----------------------------------

- OpenMP 5.0 features

  - Support relational-op != (not-equal) as one of the canonical forms of random
    access iterator.
  - Added support for mapping of the lambdas in target regions.
  - Added parsing/sema analysis for the requires directive.
  - Support nested declare target directives.
  - Make the `this` pointer implicitly mapped as `map(this[:1])`.
  - Added the `close` *map-type-modifier*.

- Various bugfixes and improvements.

New features supported for Cuda devices:

- Added support for the reductions across the teams.

- Extended number of constructs that can be executed in SPMD mode.

- Fixed support for lastprivate/reduction variables in SPMD constructs.

- New collapse clause scheme to avoid expensive remainder operations.

- New default schedule for distribute and parallel constructs.

- Simplified code generation for distribute and parallel in SPMD mode.

- Flag (``-fopenmp_optimistic_collapse``) for user to limit collapsed
  loop counter width when safe to do so.

- General performance improvement.


.. _release-notes-ubsan:

Undefined Behavior Sanitizer (UBSan)
------------------------------------

* The Implicit Conversion Sanitizer (``-fsanitize=implicit-conversion``) group
  was extended. One more type of issues is caught - implicit integer sign change.
  (``-fsanitize=implicit-integer-sign-change``).
  This makes the Implicit Conversion Sanitizer feature-complete,
  with only missing piece being bitfield handling.
  While there is a ``-Wsign-conversion`` diagnostic group that catches this kind
  of issues, it is both noisy, and does not catch **all** the cases.

  .. code-block:: c++

      bool consume(unsigned int val);

      void test(int val) {
        (void)consume(val); // If the value was negative, it is now large positive.
        (void)consume((unsigned int)val); // OK, the conversion is explicit.
      }

  Like some other ``-fsanitize=integer`` checks, these issues are **not**
  undefined behaviour. But they are not *always* intentional, and are somewhat
  hard to track down. This group is **not** enabled by ``-fsanitize=undefined``,
  but the ``-fsanitize=implicit-integer-sign-change`` check
  is enabled by ``-fsanitize=integer``.
  (as is ``-fsanitize=implicit-integer-truncation`` check)

* The Implicit Conversion Sanitizer (``-fsanitize=implicit-conversion``) has
  learned to sanitize compound assignment operators.

* ``alignment`` check has learned to sanitize the assume_aligned-like attributes:

  .. code-block:: c++

      typedef char **__attribute__((align_value(1024))) aligned_char;
      struct ac_struct {
        aligned_char a;
      };
      char **load_from_ac_struct(struct ac_struct *x) {
        return x->a; // <- check that loaded 'a' is aligned
      }

      char **passthrough(__attribute__((align_value(1024))) char **x) {
        return x; // <- check the pointer passed as function argument
      }

      char **__attribute__((alloc_align(2)))
      alloc_align(int size, unsigned long alignment);

      char **caller(int size) {
        return alloc_align(size, 1024); // <- check returned pointer
      }

      char **__attribute__((assume_aligned(1024))) get_ptr();

      char **caller2() {
        return get_ptr(); // <- check returned pointer
      }

      void *caller3(char **x) {
        return __builtin_assume_aligned(x, 1024);  // <- check returned pointer
      }

      void *caller4(char **x, unsigned long offset) {
        return __builtin_assume_aligned(x, 1024, offset);  // <- check returned pointer accounting for the offest
      }

      void process(char *data, int width) {
          #pragma omp for simd aligned(data : 1024) // <- aligned clause will be checked.
          for (int x = 0; x < width; x++)
          data[x] *= data[x];
      }


Additional Information
======================

A wide variety of additional information is available on the `Clang web
page <https://clang.llvm.org/>`_. The web page contains versions of the
API documentation which are up-to-date with the Subversion version of
the source code. You can access versions of these documents specific to
this release by going into the "``clang/docs/``" directory in the Clang
tree.

If you have any questions or comments about Clang, please feel free to
contact us via the `mailing
list <https://lists.llvm.org/mailman/listinfo/cfe-dev>`_.
