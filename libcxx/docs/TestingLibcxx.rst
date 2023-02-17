==============
Testing libc++
==============

.. contents::
  :local:

.. _testing:

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
- clang-tidy (you might need additional dev packages to compile libc++-specific clang-tidy checks)

Reproducing CI issues locally
-----------------------------

Libc++ has extensive CI that tests various configurations of the library. The testing for
all these configurations is located in ``libcxx/utils/ci/run-buildbot``. Most of our
CI jobs are being run on a Docker image for reproducibility. The definition of this Docker
image is located in ``libcxx/utils/ci/Dockerfile``. If you are looking to reproduce the
failure of a specific CI job locally, you should first drop into a Docker container that
matches our CI images by running ``libcxx/utils/ci/run-buildbot-container``, and then run
the specific CI job that you're interested in (from within the container) using the ``run-buildbot``
script above. If you want to control which compiler is used, you can set the ``CC`` and the
``CXX`` environment variables before calling ``run-buildbot`` to select the right compiler.
Take note that some CI jobs are testing the library on specific platforms and are *not* run
in our Docker image. In the general case, it is not possible to reproduce these failures
locally, unless they aren't specific to the platform.

Also note that the Docker container shares the same filesystem as your local machine, so
modifying files on your local machine will also modify what the Docker container sees.
This is useful for editing source files as you're testing your code in the Docker container.

Writing Tests
=============

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

Structure of the testing related directories
--------------------------------------------

The tests of libc++ are stored in libc++'s testing related subdirectories:

- ``libcxx/test/support`` This directory contains several helper headers with
  generic parts for the tests. The most important header is ``test_macros.h``.
  This file contains configuration information regarding the platform used.
  This is similar to the ``__config`` file in libc++'s ``include`` directory.
  Since libc++'s tests are used by other Standard libraries, tests should use
  the ``TEST_FOO`` macros instead of the ``_LIBCPP_FOO`` macros, which are
  specific to libc++.
- ``libcxx/test/std`` This directory contains the tests that validate the library under
  test conforms to the C++ Standard. The paths and the names of the test match
  the section names in the C++ Standard. Note that the C++ Standard sometimes
  reorganises its structure, therefore some tests are at a location based on
  where they appeared historically in the standard. We try to strike a balance
  between keeping things at up-to-date locations and unnecessary churn.
- ``libcxx/test/libcxx`` This directory contains the tests that validate libc++
  specific behavior and implementation details. For example, libc++ has
  "wrapped iterators" that perform bounds checks. Since those are specific to
  libc++ and not mandated by the Standard, tests for those are located under
  ``libcxx/test/libcxx``. The structure of this directories follows the
  structure of ``libcxx/test/std``.

Structure of a test
-------------------

Some platforms where libc++ is tested have requirement on the signature of
``main`` and require ``main`` to explicitly return a value. Therefore the
typical ``main`` function should look like:

.. code-block:: cpp

  int main(int, char**) {
    ...
    return 0;
  }


The C++ Standard has ``constexpr`` requirements. The typical way to test that,
is to create a helper ``test`` function that returns a ``bool`` and use the
following ``main`` function:

.. code-block:: cpp

  constexpr bool test() {
    ...
    return true;
  }

  int main(int, char**) {
    test()
    static_assert(test());

    return 0;
  }

Tests in libc++ mainly use ``assert`` and ``static_assert`` for testing. There
are a few helper macros and function that can be used to make it easier to
write common tests.

libcxx/test/support/assert_macros.h
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The header contains several macros with user specified log messages. This is
useful when a normal assertion failure lacks the information to easily
understand why the test has failed. This usually happens when the test is in a
helper function. For example the ``std::format`` tests use a helper function
for its validation. When the test fails it will give the line in the helper
function with the condition ``out == expected`` failed. Without knowing what
the value of ``format string``, ``out`` and ``expected`` are it is not easy to
understand why the test has failed. By logging these three values the point of
failure can be found without resorting to a debugger.

Several of these macros are documented to take an ``ARG``. This ``ARG``:

 - if it is a ``const char*`` or ``std::string`` its contents are written to
   the ``stderr``,
 - otherwise it must be a callable that is invoked without any additional
   arguments and is expected to produce useful output to e.g. ``stderr``.

This makes it possible to write additional information when a test fails,
either by supplying a hard-coded string or generate it at runtime.

TEST_FAIL(ARG)
^^^^^^^^^^^^^^

This macro is an unconditional failure with a log message ``ARG``. The main
use-case is to fail when code is reached that should be unreachable.


TEST_REQUIRE(CONDITION, ARG)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This macro requires its ``CONDITION`` to evaluate to ``true``. If that fails it
will fail the test with a log message ``ARG``.


TEST_LIBCPP_REQUIRE((CONDITION, ARG)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the library under test is libc++ it behaves like ``TEST_REQUIRE``, else it
is a no-op. This makes it possible to test libc++ specific behaviour. For
example testing whether the ``what()`` of an exception thrown matches libc++'s
expectations. (Usually the Standard requires certain exceptions to be thrown,
but not the contents of its ``what()`` message.)


TEST_DOES_NOT_THROW(EXPR)
^^^^^^^^^^^^^^^^^^^^^^^^^

Validates execution of ``EXPR`` does not throw an exception.

TEST_THROWS_TYPE(TYPE, EXPR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Validates the execution of ``EXPR`` throws an exception of the type ``TYPE``.


TEST_VALIDATE_EXCEPTION(TYPE, PRED, EXPR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Validates the execution of ``EXPR`` throws an exception of the type ``TYPE``
which passes validation of ``PRED``. Using this macro makes it easier to write
tests using exceptions. The code to write a test manually would be:


.. code-block:: cpp

  void test_excption([[maybe_unused]] int arg) {
  #ifndef TEST_HAS_NO_EXCEPTIONS // do nothing when tests are disabled
    try {
      foo(arg);
      assert(false); // validates foo really throws
    } catch ([[maybe_unused]] const bar& e) {
      LIBCPP_ASSERT(e.what() == what);
      return;
    }
    assert(false); // validates bar was thrown
  #endif
    }

The same test using a macro:

.. code-block:: cpp

  void test_excption([[maybe_unused]] int arg) {
    TEST_VALIDATE_EXCEPTION(bar,
                            [](const bar& e) {
                              LIBCPP_ASSERT(e.what() == what);
                            },
                            foo(arg));
    }


libcxx/test/support/concat_macros.h
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This file contains a helper macro ``TEST_WRITE_CONCATENATED`` to lazily
concatenate its arguments to a ``std::string`` and write it to ``stderr``. When
the output can't be concatenated a default message will be written to
``stderr``. This is useful for tests where the arguments use different
character types like ``char`` and ``wchar_t``, the latter can't simply be
written to ``stderrr``.

This macro is in a different header as ``assert_macros.h`` since it pulls in
additional headers.

 .. note: This macro can only be used in test using C++20 or newer. The macro
          was added at a time where most of lib++'s C++17 support was complete.
          Since it is not expected to add this to existing tests no effort was
          taken to make it work in earlier language versions.


Additional reading
------------------

The function ``CxxStandardLibraryTest`` in the file
``libcxx/utils/libcxx/test/format.py`` has documentation about writing test. It
explains the difference between the test named  ``foo.pass.cpp`` and named
``foo.verify.cpp`` are.

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
