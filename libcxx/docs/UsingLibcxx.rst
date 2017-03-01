============
Using libc++
============

.. contents::
  :local:

Getting Started
===============

If you already have libc++ installed you can use it with clang.

.. code-block:: bash

    $ clang++ -stdlib=libc++ test.cpp
    $ clang++ -std=c++11 -stdlib=libc++ test.cpp

On OS X and FreeBSD libc++ is the default standard library
and the ``-stdlib=libc++`` is not required.

.. _alternate libcxx:

If you want to select an alternate installation of libc++ you
can use the following options.

.. code-block:: bash

  $ clang++ -std=c++11 -stdlib=libc++ -nostdinc++ \
            -I<libcxx-install-prefix>/include/c++/v1 \
            -L<libcxx-install-prefix>/lib \
            -Wl,-rpath,<libcxx-install-prefix>/lib \
            test.cpp

The option ``-Wl,-rpath,<libcxx-install-prefix>/lib`` adds a runtime library
search path. Meaning that the systems dynamic linker will look for libc++ in
``<libcxx-install-prefix>/lib`` whenever the program is run. Alternatively the
environment variable ``LD_LIBRARY_PATH`` (``DYLD_LIBRARY_PATH`` on OS X) can
be used to change the dynamic linkers search paths after a program is compiled.

An example of using ``LD_LIBRARY_PATH``:

.. code-block:: bash

  $ clang++ -stdlib=libc++ -nostdinc++ \
            -I<libcxx-install-prefix>/include/c++/v1
            -L<libcxx-install-prefix>/lib \
            test.cpp -o
  $ ./a.out # Searches for libc++ in the systems library paths.
  $ export LD_LIBRARY_PATH=<libcxx-install-prefix>/lib
  $ ./a.out # Searches for libc++ along LD_LIBRARY_PATH

Using libc++experimental and ``<experimental/...>``
=====================================================

Libc++ provides implementations of experimental technical specifications
in a separate library, ``libc++experimental.a``. Users of ``<experimental/...>``
headers may be required to link ``-lc++experimental``.

.. code-block:: bash

  $ clang++ -std=c++14 -stdlib=libc++ test.cpp -lc++experimental

Libc++experimental.a may not always be available, even when libc++ is already
installed. For information on building libc++experimental from source see
:ref:`Building Libc++ <build instructions>` and
:ref:`libc++experimental CMake Options <libc++experimental options>`.

Also see the `Experimental Library Implementation Status <http://libcxx.llvm.org/ts1z_status.html>`__
page.

.. warning::
  Experimental libraries are Experimental.
    * The contents of the ``<experimental/...>`` headers and ``libc++experimental.a``
      library will not remain compatible between versions.
    * No guarantees of API or ABI stability are provided.

Using libc++ on Linux
=====================

On Linux libc++ can typically be used with only '-stdlib=libc++'. However
some libc++ installations require the user manually link libc++abi themselves.
If you are running into linker errors when using libc++ try adding '-lc++abi'
to the link line.  For example:

.. code-block:: bash

  $ clang++ -stdlib=libc++ test.cpp -lc++ -lc++abi -lm -lc -lgcc_s -lgcc

Alternately, you could just add libc++abi to your libraries list, which in
most situations will give the same result:

.. code-block:: bash

  $ clang++ -stdlib=libc++ test.cpp -lc++abi


Using libc++ with GCC
---------------------

GCC does not provide a way to switch from libstdc++ to libc++. You must manually
configure the compile and link commands.

In particular you must tell GCC to remove the libstdc++ include directories
using ``-nostdinc++`` and to not link libstdc++.so using ``-nodefaultlibs``.

Note that ``-nodefaultlibs`` removes all of the standard system libraries and
not just libstdc++ so they must be manually linked. For example:

.. code-block:: bash

  $ g++ -nostdinc++ -I<libcxx-install-prefix>/include/c++/v1 \
         test.cpp -nodefaultlibs -lc++ -lc++abi -lm -lc -lgcc_s -lgcc


GDB Pretty printers for libc++
------------------------------

GDB does not support pretty-printing of libc++ symbols by default. Unfortunately
libc++ does not provide pretty-printers itself. However there are 3rd
party implementations available and although they are not officially
supported by libc++ they may be useful to users.

Known 3rd Party Implementations Include:

* `Koutheir's libc++ pretty-printers <https://github.com/koutheir/libcxx-pretty-printers>`_.


Libc++ Configuration Macros
===========================

Libc++ provides a number of configuration macros which can be used to enable
or disable extended libc++ behavior, including enabling "debug mode" or
thread safety annotations.

**_LIBCPP_DEBUG**:
  See :ref:`using-debug-mode` for more information.

**_LIBCPP_ENABLE_THREAD_SAFETY_ANNOTATIONS**:
  This macro is used to enable -Wthread-safety annotations on libc++'s
  ``std::mutex`` and ``std::lock_guard``. By default these annotations are
  disabled and must be manually enabled by the user.

**_LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS**:
  This macro is used to disable all visibility annotations inside libc++.
  Defining this macro and then building libc++ with hidden visibility gives a
  build of libc++ which does not export any symbols, which can be useful when
  building statically for inclusion into another library.

**_LIBCPP_ENABLE_TUPLE_IMPLICIT_REDUCED_ARITY_EXTENSION**:
  This macro is used to re-enable an extension in `std::tuple` which allowed
  it to be implicitly constructed from fewer initializers than contained
  elements. Elements without an initializer are default constructed. For example:

  .. code-block:: cpp

    std::tuple<std::string, int, std::error_code> foo() {
      return {"hello world", 42}; // default constructs error_code
    }


  Since libc++ 4.0 this extension has been disabled by default. This macro
  may be defined to re-enable it in order to support existing code that depends
  on the extension. New use of this extension should be discouraged.
  See `PR 27374 <http://llvm.org/PR27374>`_ for more information.

  Note: The "reduced-arity-initialization" extension is still offered but only
  for explicit conversions. Example:

  .. code-block:: cpp

    auto foo() {
      using Tup = std::tuple<std::string, int, std::error_code>;
      return Tup{"hello world", 42}; // explicit constructor called. OK.
    }

**_LIBCPP_DISABLE_ADDITIONAL_DIAGNOSTICS**:
  This macro disables the additional diagnostics generated by libc++ using the
  `diagnose_if` attribute. These additional diagnostics include checks for:

    * Giving `set`, `map`, `multiset`, `multimap` a comparator which is not
      const callable.

C++17 Specific Configuration Macros
-----------------------------------
**_LIBCPP_ENABLE_CXX17_REMOVED_FEATURES**:
  This macro is used to re-enable all the features removed in C++17. The effect
  is equivalent to manually defining each macro listed below.

**_LIBCPP_ENABLE_CXX17_REMOVED_UNEXPECTED_FUNCTIONS**:
  This macro is used to re-enable the `set_unexpected`, `get_unexpected`, and
  `unexpected` functions, which were removed in C++17.

**_LIBCPP_ENABLE_CXX17_REMOVED_AUTO_PTR**:
  This macro is used to re-enable `std::auto_ptr` in C++17.
