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
``cxx-test-depends`` target. For example:

.. code-block:: bash

  $ cd <monorepo-root>
  $ make -C <build> cxx-test-depends # If you want to make sure the targets get rebuilt
  $ <build>/bin/llvm-lit -sv libcxx/test/std/re # Run all of the std::regex tests
  $ <build>/bin/llvm-lit -sv libcxx/test/std/depr/depr.c.headers/stdlib_h.pass.cpp # Run a single test
  $ <build>/bin/llvm-lit -sv libcxx/test/std/atomics libcxx/test/std/threads # Test std::thread and std::atomic

.. note::
  If you used the Bootstrapping build instead of the default runtimes build, the
  ``cxx-test-depends`` target is instead named ``runtimes-test-depends``, and
  you will need to prefix ``<build>/runtimes/runtimes-<target>-bins/`` to the
  paths of all tests.

In the default configuration, the tests are built against headers that form a
fake installation root of libc++. This installation root has to be updated when
changes are made to the headers, so you should re-run the ``cxx-test-depends``
target before running the tests manually with ``lit`` when you make any sort of
change, including to the headers.

Sometimes you'll want to change the way LIT is running the tests. Custom options
can be specified using the ``--param <name>=<val>`` flag. The most common option
you'll want to change is the standard dialect (ie ``-std=c++XX``). By default the
test suite will select the newest C++ dialect supported by the compiler and use
that. However, you can manually specify the option like so if you want:

.. code-block:: bash

  $ <build>/bin/llvm-lit -sv libcxx/test/std/containers # Run the tests with the newest -std
  $ <build>/bin/llvm-lit -sv libcxx/test/std/containers --param std=c++03 # Run the tests in C++03

Other parameters are supported by the test suite. Those are defined in ``libcxx/utils/libcxx/test/params.py``.
If you want to customize how to run the libc++ test suite beyond what is available
in ``params.py``, you most likely want to use a custom site configuration instead.

The libc++ test suite works by loading a site configuration that defines various
"base" parameters (via Lit substitutions). These base parameters represent things
like the compiler to use for running the tests, which default compiler and linker
flags to use, and how to run an executable. This system is meant to be easily
extended for custom needs, in particular when porting the libc++ test suite to
new platforms.

Using a Custom Site Configuration
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

Additional tools
----------------

The libc++ test suite uses a few optional tools to improve the code quality.

These tools are:
- clang-query
- clang-tidy

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
  $ ninja cxx-benchmarks

This will build all of the benchmarks under ``<libcxx-src>/benchmarks`` to be
built against the just-built libc++. The compiled tests are output into
``build/projects/libcxx/benchmarks``.

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

  $ cd build/projects/libcxx/benchmarks
  $ ./algorithms.libcxx.out # Runs all the benchmarks
  $ ./algorithms.libcxx.out --benchmark_filter=BM_Sort.* # Only runs the sort benchmarks

For more information about running benchmarks see `Google Benchmark`_.
