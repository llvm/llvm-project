Testing
=======

The LLDB test suite consists of Python scripts located under the test
directory. Each script contains a number of test cases and is usually
accompanied by a C (C++, ObjC, etc.) source file. Each test first compiles the
source file and then uses LLDB to debug the resulting executable. The tests
verify both the LLDB command line interface and the scripting API.

.. contents::
   :local:

.. note::

   On Windows any invocations of python should be replaced with python_d, the
   debug interpreter, when running the test suite against a debug version of
   LLDB.

.. note::

   On NetBSD you must export ``LD_LIBRARY_PATH=$PWD/lib`` in your environment.
   This is due to lack of the ``$ORIGIN`` linker feature.

Running the Full Test Suite
---------------------------

The easiest way to run the LLDB test suite is to use the ``check-lldb`` build
target.

By default, the ``check-lldb`` target builds the test programs with the same
compiler that was used to build LLDB. To build the tests with a different
compiler, you can set the ``LLDB_TEST_C_COMPILER`` or the
``LLDB_TEST_CXX_COMPILER`` CMake variables. These variables are ignored unless
the respective ``LLDB_TEST_USE_CUSTOM_C_COMPILER`` and
``LLDB_TEST_USE_CUSTOM_CXX_COMPILER`` are set to ``ON``.

It is possible to customize the architecture of the test binaries and compiler
used by appending ``-A`` and ``-C`` options respectively to the CMake variable
``LLDB_TEST_USER_ARGS``. For example, to test LLDB against 32-bit binaries
built with a custom version of clang, do:

::

   > cmake -DLLDB_TEST_USER_ARGS="-A i386 -C /path/to/custom/clang" -G Ninja
   > ninja check-lldb

Note that multiple ``-A`` and ``-C`` flags can be specified to
``LLDB_TEST_USER_ARGS``.


Running a Specific Test or Set of Tests
---------------------------------------

In addition to running all the LLDB test suites with the ``check-lldb`` CMake
target above, it is possible to run individual LLDB tests. If you have a CMake
build you can use the ``lldb-dotest`` binary, which is a wrapper around
``dotest.py`` that passes all the arguments configured by CMake. Alternatively,
you can use ``dotest.py`` directly, if you want to run a test one-off with a
different configuration.


For example, to run the test cases defined in TestInferiorCrashing.py, run:

::

   > lldb-dotest -p TestInferiorCrashing.py

::

   > cd $lldb/test
   > python dotest.py --executable <path-to-lldb> -p TestInferiorCrashing.py ../packages/Python/lldbsuite/test

If the test is not specified by name (e.g. if you leave the ``-p`` argument
off),  all tests in that directory will be executed:


::

   > lldb-dotest functionalities/data-formatter

::

   > python dotest.py --executable <path-to-lldb> functionalities/data-formatter

Many more options that are available. To see a list of all of them, run:

::

   > python dotest.py -h

The ``dotest.py`` script runs tests in parallel by default. To disable the
parallel test running feature, use the ``--no-multiprocess`` flag. The number
of concurrent tests is controlled by the ``LLDB_TEST_THREADS`` environment
variable or the ``--threads command`` line parameter. The default value is the
number of CPU cores on your system.

The parallel test running feature will handle an additional ``--test-subdir
SUBDIR`` arg. When specified, ``SUBDIR`` is relative to the root test directory
and will limit all parallel test running to that subdirectory's tree of tests.

The parallel test runner will run all tests within a given directory serially,
but will run multiple directories concurrently. Thus, as a test writer, we
provide serialized test run semantics within a directory. Note child
directories are considered entirely separate, so two child directories could be
running in parallel with a parent directory.

Running the Test Suite Remotely
-------------------------------

Running the test-suite remotely is similar to the process of running a local
test suite, but there are two things to have in mind:

1. You must have the lldb-server running on the remote system, ready to accept
   multiple connections. For more information on how to setup remote debugging
   see the Remote debugging page.
2. You must tell the test-suite how to connect to the remote system. This is
   achieved using the ``--platform-name``, ``--platform-url`` and
   ``--platform-working-dir`` parameters to ``dotest.py``. These parameters
   correspond to the platform select and platform connect LLDB commands. You
   will usually also need to specify the compiler and architecture for the
   remote system.

Currently, running the remote test suite is supported only with ``dotest.py`` (or
dosep.py with a single thread), but we expect this issue to be addressed in the
near future.

Debugging Test Failures
-----------------------

On non-Windows platforms, you can use the ``-d`` option to ``dotest.py`` which
will cause the script to wait for a while until a debugger is attached.

Debugging Test Failures on Windows
----------------------------------

On Windows, it is strongly recommended to use Python Tools for Visual Studio
for debugging test failures. It can seamlessly step between native and managed
code, which is very helpful when you need to step through the test itself, and
then into the LLDB code that backs the operations the test is performing.

A quick guide to getting started with PTVS is as follows:

#. Install PTVS
#. Create a Visual Studio Project for the Python code.
    #. Go to File -> New -> Project -> Python -> From Existing Python Code.
    #. Choose llvm/tools/lldb as the directory containing the Python code.
    #. When asked where to save the .pyproj file, choose the folder ``llvm/tools/lldb/pyproj``. This is a special folder that is ignored by the ``.gitignore`` file, since it is not checked in.
#. Set test/dotest.py as the startup file
#. Make sure there is a Python Environment installed for your distribution. For example, if you installed Python to ``C:\Python35``, PTVS needs to know that this is the interpreter you want to use for running the test suite.
    #. Go to Tools -> Options -> Python Tools -> Environment Options
    #. Click Add Environment, and enter Python 3.5 Debug for the name. Fill out the values correctly.
#. Configure the project to use this debug interpreter.
    #. Right click the Project node in Solution Explorer.
    #. In the General tab, Make sure Python 3.5 Debug is the selected Interpreter.
    #. In Debug/Search Paths, enter the path to your ninja/lib/site-packages directory.
    #. In Debug/Environment Variables, enter ``VCINSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\``.
    #. If you want to enabled mixed mode debugging, check Enable native code debugging (this slows down debugging, so enable it only on an as-needed basis.)
#. Set the command line for the test suite to run.
    #. Right click the project in solution explorer and choose the Debug tab.
    #. Enter the arguments to dotest.py. Note you must add --no-multiprocess
    #. Example command options:

::

   # quiet mode
   -q
   --arch=i686
   # Path to debug lldb.exe
   --executable D:/src/llvmbuild/ninja/bin/lldb.exe
   # Directory to store log files
   -s D:/src/llvmbuild/ninja/lldb-test-traces
   -u CXXFLAGS -u CFLAGS
   # If a test crashes, show JIT debugging dialog.
   --enable-crash-dialog
   # Path to release clang.exe
   -C d:\src\llvmbuild\ninja_release\bin\clang.exe
   # Path to the particular test you want to debug.
   -p TestPaths.py
   # Root of test tree
   D:\src\llvm\tools\lldb\packages\Python\lldbsuite\test
   # Required in order to be able to debug the test.
   --no-multiprocess

::

   -q --arch=i686 --executable D:/src/llvmbuild/ninja/bin/lldb.exe -s D:/src/llvmbuild/ninja/lldb-test-traces -u CXXFLAGS -u CFLAGS --enable-crash-dialog -C d:\src\llvmbuild\ninja_release\bin\clang.exe -p TestPaths.py D:\src\llvm\tools\lldb\packages\Python\lldbsuite\test --no-multiprocess



