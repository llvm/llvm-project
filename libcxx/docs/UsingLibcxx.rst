.. _using-libcxx:

============
Using libc++
============

.. contents::
  :local:

Usually, libc++ is packaged and shipped by a vendor through some delivery vehicle
(operating system distribution, SDK, toolchain, etc) and users don't need to do
anything special in order to use the library.

This page contains information about configuration knobs that can be used by
users when they know libc++ is used by their toolchain, and how to use libc++
when it is not the default library used by their toolchain.


Using a different version of the C++ Standard
=============================================

Libc++ implements the various versions of the C++ Standard. Changing the version of
the standard can be done by passing ``-std=c++XY`` to the compiler. Libc++ will
automatically detect what Standard is being used and will provide functionality that
matches that Standard in the library.

.. code-block:: bash

  $ clang++ -std=c++17 test.cpp

.. warning::
  Using ``-std=c++XY`` with a version of the Standard that has not been ratified yet
  is considered unstable. Libc++ reserves the right to make breaking changes to the
  library until the standard has been ratified.


Using libc++experimental and ``<experimental/...>``
===================================================

Libc++ provides implementations of experimental technical specifications
in a separate library, ``libc++experimental.a``. Users of ``<experimental/...>``
headers may be required to link ``-lc++experimental``. Note that not all
vendors ship ``libc++experimental.a``, and as a result, you may not be
able to use those experimental features.

.. code-block:: bash

  $ clang++ test.cpp -lc++experimental

.. warning::
  Experimental libraries are Experimental.
    * The contents of the ``<experimental/...>`` headers and ``libc++experimental.a``
      library will not remain compatible between versions.
    * No guarantees of API or ABI stability are provided.
    * When the standardized version of an experimental feature is implemented,
      the experimental feature is removed two releases after the non-experimental
      version has shipped. The full policy is explained :ref:`here <experimental features>`.


Using libc++ when it is not the system default
==============================================

On systems where libc++ is provided but is not the default, Clang provides a flag
called ``-stdlib=`` that can be used to decide which standard library is used.
Using ``-stdlib=libc++`` will select libc++:

.. code-block:: bash

  $ clang++ -stdlib=libc++ test.cpp

On systems where libc++ is the library in use by default such as macOS and FreeBSD,
this flag is not required.


.. _alternate libcxx:

Using a custom built libc++
===========================

Most compilers provide a way to disable the default behavior for finding the
standard library and to override it with custom paths. With Clang, this can
be done with:

.. code-block:: bash

  $ clang++ -nostdinc++ -nostdlib++           \
            -isystem <install>/include/c++/v1 \
            -L <install>/lib                  \
            -Wl,-rpath,<install>/lib          \
            -lc++                             \
            test.cpp

The option ``-Wl,-rpath,<install>/lib`` adds a runtime library search path,
which causes the system's dynamic linker to look for libc++ in ``<install>/lib``
whenever the program is loaded.

GCC does not support the ``-nostdlib++`` flag, so one must use ``-nodefaultlibs``
instead. Since that removes all the standard system libraries and not just libc++,
the system libraries must be re-added manually. For example:

.. code-block:: bash

  $ g++ -nostdinc++ -nodefaultlibs           \
        -isystem <install>/include/c++/v1    \
        -L <install>/lib                     \
        -Wl,-rpath,<install>/lib             \
        -lc++ -lc++abi -lm -lc -lgcc_s -lgcc \
        test.cpp


GDB Pretty printers for libc++
==============================

GDB does not support pretty-printing of libc++ symbols by default. However, libc++ does
provide pretty-printers itself. Those can be used as:

.. code-block:: bash

  $ gdb -ex "source <libcxx>/utils/gdb/libcxx/printers.py" \
        -ex "python register_libcxx_printer_loader()" \
        <args>


Libc++ Configuration Macros
===========================

Libc++ provides a number of configuration macros which can be used to enable
or disable extended libc++ behavior, including enabling "debug mode" or
thread safety annotations.

**_LIBCPP_DEBUG**:
  See :ref:`using-debug-mode` for more information.

**_LIBCPP_ENABLE_THREAD_SAFETY_ANNOTATIONS**:
  This macro is used to enable -Wthread-safety annotations on libc++'s
  ``std::mutex`` and ``std::lock_guard``. By default, these annotations are
  disabled and must be manually enabled by the user.

**_LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS**:
  This macro is used to disable all visibility annotations inside libc++.
  Defining this macro and then building libc++ with hidden visibility gives a
  build of libc++ which does not export any symbols, which can be useful when
  building statically for inclusion into another library.

**_LIBCPP_DISABLE_EXTERN_TEMPLATE**:
  This macro is used to disable extern template declarations in the libc++
  headers. The intended use case is for clients who wish to use the libc++
  headers without taking a dependency on the libc++ library itself.

**_LIBCPP_DISABLE_ADDITIONAL_DIAGNOSTICS**:
  This macro disables the additional diagnostics generated by libc++ using the
  `diagnose_if` attribute. These additional diagnostics include checks for:

    * Giving `set`, `map`, `multiset`, `multimap` and their `unordered_`
      counterparts a comparator which is not const callable.
    * Giving an unordered associative container a hasher that is not const
      callable.

**_LIBCPP_NO_VCRUNTIME**:
  Microsoft's C and C++ headers are fairly entangled, and some of their C++
  headers are fairly hard to avoid. In particular, `vcruntime_new.h` gets pulled
  in from a lot of other headers and provides definitions which clash with
  libc++ headers, such as `nothrow_t` (note that `nothrow_t` is a struct, so
  there's no way for libc++ to provide a compatible definition, since you can't
  have multiple definitions).

  By default, libc++ solves this problem by deferring to Microsoft's vcruntime
  headers where needed. However, it may be undesirable to depend on vcruntime
  headers, since they may not always be available in cross-compilation setups,
  or they may clash with other headers. The `_LIBCPP_NO_VCRUNTIME` macro
  prevents libc++ from depending on vcruntime headers. Consequently, it also
  prevents libc++ headers from being interoperable with vcruntime headers (from
  the aforementioned clashes), so users of this macro are promising to not
  attempt to combine libc++ headers with the problematic vcruntime headers. This
  macro also currently prevents certain `operator new`/`operator delete`
  replacement scenarios from working, e.g. replacing `operator new` and
  expecting a non-replaced `operator new[]` to call the replaced `operator new`.

**_LIBCPP_ENABLE_NODISCARD**:
  Allow the library to add ``[[nodiscard]]`` attributes to entities not specified
  as ``[[nodiscard]]`` by the current language dialect. This includes
  backporting applications of ``[[nodiscard]]`` from newer dialects and
  additional extended applications at the discretion of the library. All
  additional applications of ``[[nodiscard]]`` are disabled by default.
  See :ref:`Extended Applications of [[nodiscard]] <nodiscard extension>` for
  more information.

**_LIBCPP_DISABLE_NODISCARD_EXT**:
  This macro prevents the library from applying ``[[nodiscard]]`` to entities
  purely as an extension. See :ref:`Extended Applications of [[nodiscard]] <nodiscard extension>`
  for more information.

**_LIBCPP_DISABLE_DEPRECATION_WARNINGS**:
  This macro disables warnings when using deprecated components. For example,
  using `std::auto_ptr` when compiling in C++11 mode will normally trigger a
  warning saying that `std::auto_ptr` is deprecated. If the macro is defined,
  no warning will be emitted. By default, this macro is not defined.

C++17 Specific Configuration Macros
-----------------------------------
**_LIBCPP_ENABLE_CXX17_REMOVED_FEATURES**:
  This macro is used to re-enable all the features removed in C++17. The effect
  is equivalent to manually defining each macro listed below.

**_LIBCPP_ENABLE_CXX17_REMOVED_AUTO_PTR**:
  This macro is used to re-enable `auto_ptr`.

**_LIBCPP_ENABLE_CXX17_REMOVED_BINDERS**:
  This macro is used to re-enable the `binder1st`, `binder2nd`,
  `pointer_to_unary_function`, `pointer_to_binary_function`, `mem_fun_t`,
  `mem_fun1_t`, `mem_fun_ref_t`, `mem_fun1_ref_t`, `const_mem_fun_t`,
  `const_mem_fun1_t`, `const_mem_fun_ref_t`, and `const_mem_fun1_ref_t`
  class templates, and the `bind1st`, `bind2nd`, `mem_fun`, `mem_fun_ref`,
  and `ptr_fun` functions.

**_LIBCPP_ENABLE_CXX17_REMOVED_RANDOM_SHUFFLE**:
  This macro is used to re-enable the `random_shuffle` algorithm.

**_LIBCPP_ENABLE_CXX17_REMOVED_UNEXPECTED_FUNCTIONS**:
  This macro is used to re-enable `set_unexpected`, `get_unexpected`, and
  `unexpected`.

C++20 Specific Configuration Macros:
------------------------------------
**_LIBCPP_DISABLE_NODISCARD_AFTER_CXX17**:
  This macro can be used to disable diagnostics emitted from functions marked
  ``[[nodiscard]]`` in dialects after C++17.  See :ref:`Extended Applications of [[nodiscard]] <nodiscard extension>`
  for more information.

**_LIBCPP_ENABLE_CXX20_REMOVED_FEATURES**:
  This macro is used to re-enable all the features removed in C++20. The effect
  is equivalent to manually defining each macro listed below.

**_LIBCPP_ENABLE_CXX20_REMOVED_ALLOCATOR_MEMBERS**:
  This macro is used to re-enable redundant members of `allocator<T>`,
  including `pointer`, `reference`, `rebind`, `address`, `max_size`,
  `construct`, `destroy`, and the two-argument overload of `allocate`.

**_LIBCPP_ENABLE_CXX20_REMOVED_BINDER_TYPEDEFS**:
  This macro is used to re-enable the `argument_type`, `result_type`,
  `first_argument_type`, and `second_argument_type` members of class
  templates such as `plus`, `logical_not`, `hash`, and `owner_less`.

**_LIBCPP_ENABLE_CXX20_REMOVED_NEGATORS**:
  This macro is used to re-enable `not1`, `not2`, `unary_negate`,
  and `binary_negate`.

**_LIBCPP_ENABLE_CXX20_REMOVED_RAW_STORAGE_ITERATOR**:
  This macro is used to re-enable `raw_storage_iterator`.

**_LIBCPP_ENABLE_CXX20_REMOVED_TYPE_TRAITS**:
  This macro is used to re-enable `is_literal_type`, `is_literal_type_v`,
  `result_of` and `result_of_t`.


Libc++ Extensions
=================

This section documents various extensions provided by libc++, how they're
provided, and any information regarding how to use them.

.. _nodiscard extension:

Extended applications of ``[[nodiscard]]``
------------------------------------------

The ``[[nodiscard]]`` attribute is intended to help users find bugs where
function return values are ignored when they shouldn't be. After C++17 the
C++ standard has started to declared such library functions as ``[[nodiscard]]``.
However, this application is limited and applies only to dialects after C++17.
Users who want help diagnosing misuses of STL functions may desire a more
liberal application of ``[[nodiscard]]``.

For this reason libc++ provides an extension that does just that! The
extension must be enabled by defining ``_LIBCPP_ENABLE_NODISCARD``. The extended
applications of ``[[nodiscard]]`` takes two forms:

1. Backporting ``[[nodiscard]]`` to entities declared as such by the
   standard in newer dialects, but not in the present one.

2. Extended applications of ``[[nodiscard]]``, at the library's discretion,
   applied to entities never declared as such by the standard.

Users may also opt-out of additional applications ``[[nodiscard]]`` using
additional macros.

Applications of the first form, which backport ``[[nodiscard]]`` from a newer
dialect, may be disabled using macros specific to the dialect in which it was
added. For example, ``_LIBCPP_DISABLE_NODISCARD_AFTER_CXX17``.

Applications of the second form, which are pure extensions, may be disabled
by defining ``_LIBCPP_DISABLE_NODISCARD_EXT``.


Entities declared with ``_LIBCPP_NODISCARD_EXT``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section lists all extended applications of ``[[nodiscard]]`` to entities
which no dialect declares as such (See the second form described above).

* ``adjacent_find``
* ``all_of``
* ``any_of``
* ``binary_search``
* ``clamp``
* ``count_if``
* ``count``
* ``equal_range``
* ``equal``
* ``find_end``
* ``find_first_of``
* ``find_if_not``
* ``find_if``
* ``find``
* ``get_temporary_buffer``
* ``includes``
* ``is_heap_until``
* ``is_heap``
* ``is_partitioned``
* ``is_permutation``
* ``is_sorted_until``
* ``is_sorted``
* ``lexicographical_compare``
* ``lower_bound``
* ``max_element``
* ``max``
* ``min_element``
* ``min``
* ``minmax_element``
* ``minmax``
* ``mismatch``
* ``none_of``
* ``remove_if``
* ``remove``
* ``search_n``
* ``search``
* ``unique``
* ``upper_bound``
* ``lock_guard``'s constructors
* ``as_const``
* ``forward``
* ``move``
* ``move_if_noexcept``
* ``identity::operator()``
* ``to_integer``
* ``to_underlying``
