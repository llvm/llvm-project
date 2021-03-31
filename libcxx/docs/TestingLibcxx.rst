==============
Testing libc++
==============

.. contents::
  :local:

Getting Started
===============

libc++ uses LIT to configure and run its tests.

The primary way to run the libc++ tests is by using ``make check-cxx``.

However since libc++ can be used in any number of possible
configurations it is important to customize the way LIT builds and runs
the tests. This guide provides information on how to use LIT directly to
test libc++.

Please see the `Lit Command Guide`_ for more information about LIT.

.. _LIT Command Guide: https://llvm.org/docs/CommandGuide/lit.html

Usage
-----

After building libc++, you can run parts of the libc++ test suite by simply
running ``llvm-lit`` on a specified test or directory. If you're unsure
whether the required libraries have been built, you can use the
`cxx-test-depends` target. For example:

.. code-block:: bash

  $ cd <monorepo-root>
  $ make -C <build> cxx-test-depends # If you want to make sure the targets get rebuilt
  $ <build>/bin/llvm-lit -sv libcxx/test/std/re # Run all of the std::regex tests
  $ <build>/bin/llvm-lit -sv libcxx/test/std/depr/depr.c.headers/stdlib_h.pass.cpp # Run a single test
  $ <build>/bin/llvm-lit -sv libcxx/test/std/atomics libcxx/test/std/threads # Test std::thread and std::atomic

In the default configuration, the tests are built against headers that form a
fake installation root of libc++. This installation root has to be updated when
changes are made to the headers, so you should re-run the `cxx-test-depends`
target before running the tests manually with `lit` when you make any sort of
change, including to the headers.

Sometimes you'll want to change the way LIT is running the tests. Custom options
can be specified using the `--param=<name>=<val>` flag. The most common option
you'll want to change is the standard dialect (ie -std=c++XX). By default the
test suite will select the newest C++ dialect supported by the compiler and use
that. However if you want to manually specify the option like so:

.. code-block:: bash

  $ <build>/bin/llvm-lit -sv libcxx/test/std/containers # Run the tests with the newest -std
  $ <build>/bin/llvm-lit -sv libcxx/test/std/containers --param=std=c++03 # Run the tests in C++03

Occasionally you'll want to add extra compile or link flags when testing.
You can do this as follows:

.. code-block:: bash

  $ <build>/bin/llvm-lit -sv libcxx/test --param=compile_flags='-Wcustom-warning'
  $ <build>/bin/llvm-lit -sv libcxx/test --param=link_flags='-L/custom/library/path'

Some other common examples include:

.. code-block:: bash

  # Specify a custom compiler.
  $ <build>/bin/llvm-lit -sv libcxx/test/std --param=cxx_under_test=/opt/bin/g++

  # Disable warnings in the test suite
  $ <build>/bin/llvm-lit -sv libcxx/test --param=enable_warnings=False

  # Use UBSAN when running the tests.
  $ <build>/bin/llvm-lit -sv libcxx/test --param=use_sanitizer=Undefined

Using a custom site configuration
---------------------------------

By default, the libc++ test suite will use a site configuration that matches
the current CMake configuration. It does so by generating a ``lit.site.cfg``
file in the build directory from one of the configuration file templates in
``libcxx/test/configs/``, and pointing ``llvm-lit`` (which is a wrapper around
``llvm/utils/lit/lit.py``) to that file. So when you're running
``<build>/bin/llvm-lit``, the generated ``lit.site.cfg`` file is always loaded
instead of ``libcxx/test/lit.cfg.py``. If you want to use a custom site
configuration, simply point the CMake build to it using
``-DLIBCXX_TEST_CONFIG=<path-to-site-config>``, and that site configuration
will be used instead. That file can use CMake variables inside it to make
configuration easier.

   .. code-block:: bash

     $ cmake <options> -DLIBCXX_TEST_CONFIG=<path-to-site-config>
     $ make -C <build> cxx-test-depends
     $ <build>/bin/llvm-lit -sv libcxx/test # will use your custom config file


LIT Options
===========

:program:`lit` [*options*...] [*filenames*...]

Command Line Options
--------------------

To use these options you pass them on the LIT command line as ``--param NAME``
or ``--param NAME=VALUE``. Some options have default values specified during
CMake's configuration. Passing the option on the command line will override the
default.

.. program:: lit

.. option:: cxx_under_test=<path/to/compiler>

  Specify the compiler used to build the tests.

.. option:: stdlib=<stdlib name>

  **Values**: libc++, libstdc++, msvc

  Specify the C++ standard library being tested. The default is libc++ if this
  option is not provided. This option is intended to allow running the libc++
  test suite against other standard library implementations.

.. option:: std=<standard version>

  **Values**: c++03, c++11, c++14, c++17, c++2a, c++2b

  Change the standard version used when building the tests.

.. option:: cxx_headers=<path/to/headers>

  Specify the c++ standard library headers that are tested. By default the
  headers in the source tree are used.

.. option:: cxx_library_root=<path/to/lib/>

  Specify the directory of the libc++ library to be tested. By default the
  library folder of the build directory is used.


.. option:: cxx_runtime_root=<path/to/lib/>

  Specify the directory of the libc++ library to use at runtime. This directory
  is not added to the linkers search path. This can be used to compile tests
  against one version of libc++ and run them using another. The default value
  for this option is `cxx_library_root`.

.. option:: use_system_cxx_lib=<bool>

  **Default**: False

  Enable or disable testing against the installed version of libc++ library.
  This impacts whether the ``with_system_cxx_lib`` Lit feature is defined or
  not. The ``cxx_library_root`` and ``cxx_runtime_root`` parameters should
  still be used to specify the path of the library to link to and run against,
  respectively.

.. option:: debug_level=<level>

  **Values**: 0, 1

  Enable the use of debug mode. Level 0 enables assertions and level 1 enables
  assertions and debugging of iterator misuse.

.. option:: use_sanitizer=<sanitizer name>

  **Values**: Memory, MemoryWithOrigins, Address, Undefined

  Run the tests using the given sanitizer. If LLVM_USE_SANITIZER was given when
  building libc++ then that sanitizer will be used by default.

.. option:: llvm_unwinder

  Enable the use of LLVM unwinder instead of libgcc.

.. option:: builtins_library

  Path to the builtins library to use instead of libgcc.


Writing Tests
-------------

When writing tests for the libc++ test suite, you should follow a few guidelines.
This will ensure that your tests can run on a wide variety of hardware and under
a wide variety of configurations. We have several unusual configurations such as
building the tests on one host but running them on a different host, which add a
few requirements to the test suite. Here's some stuff you should know:

- All tests are run in a temporary directory that is unique to that test and
  cleaned up after the test is done.
- When a test needs data files as inputs, these data files can be saved in the
  repository (when reasonable) and referenced by the test as
  ``// FILE_DEPENDENCIES: <path-to-dependencies>``. Copies of these files or
  directories will be made available to the test in the temporary directory
  where it is run.
- You should never hardcode a path from the build-host in a test, because that
  path will not necessarily be available on the host where the tests are run.
- You should try to reduce the runtime dependencies of each test to the minimum.
  For example, requiring Python to run a test is bad, since Python is not
  necessarily available on all devices we may want to run the tests on (even
  though supporting Python is probably trivial for the build-host).

Benchmarks
==========

Libc++ contains benchmark tests separately from the test of the test suite.
The benchmarks are written using the `Google Benchmark`_ library, a copy of which
is stored in the libc++ repository.

For more information about using the Google Benchmark library see the
`official documentation <https://github.com/google/benchmark>`_.

.. _`Google Benchmark`: https://github.com/google/benchmark

Building Benchmarks
-------------------

The benchmark tests are not built by default. The benchmarks can be built using
the ``cxx-benchmarks`` target.

An example build would look like:

.. code-block:: bash

  $ cd build
  $ cmake [options] <path to libcxx sources>
  $ make cxx-benchmarks

This will build all of the benchmarks under ``<libcxx-src>/benchmarks`` to be
built against the just-built libc++. The compiled tests are output into
``build/benchmarks``.

The benchmarks can also be built against the platforms native standard library
using the ``-DLIBCXX_BUILD_BENCHMARKS_NATIVE_STDLIB=ON`` CMake option. This
is useful for comparing the performance of libc++ to other standard libraries.
The compiled benchmarks are named ``<test>.libcxx.out`` if they test libc++ and
``<test>.native.out`` otherwise.

Also See:

  * :ref:`Building Libc++ <build instructions>`
  * :ref:`CMake Options`

Running Benchmarks
------------------

The benchmarks must be run manually by the user. Currently there is no way
to run them as part of the build.

For example:

.. code-block:: bash

  $ cd build/benchmarks
  $ make cxx-benchmarks
  $ ./algorithms.libcxx.out # Runs all the benchmarks
  $ ./algorithms.libcxx.out --benchmark_filter=BM_Sort.* # Only runs the sort benchmarks

For more information about running benchmarks see `Google Benchmark`_.
