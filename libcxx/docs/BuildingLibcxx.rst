.. _BuildingLibcxx:

===============
Building libc++
===============

.. contents::
  :local:

.. _build instructions:

Getting Started
===============

On Mac OS 10.7 (Lion) and later, the easiest way to get this library is to install
Xcode 4.2 or later.  However if you want to install tip-of-trunk from here
(getting the bleeding edge), read on.

The following instructions describe how to checkout, build, test and
(optionally) install libc++ and libc++abi.

If your system already provides a libc++ installation it is important to be
careful not to replace it. Remember Use the CMake option
``CMAKE_INSTALL_PREFIX`` to select a safe place to install libc++.

.. warning::
  * Replacing your systems libc++ installation could render the system non-functional.
  * macOS will not boot without a valid copy of ``libc++.1.dylib`` in ``/usr/lib``.

.. code-block:: bash

  $ git clone https://github.com/llvm/llvm-project.git
  $ cd llvm-project
  $ mkdir build && cd build
  $ cmake -DCMAKE_C_COMPILER=clang \
          -DCMAKE_CXX_COMPILER=clang++ \
          -DLLVM_ENABLE_PROJECTS="libcxx;libcxxabi" \
          ../llvm
  $ make # Build
  $ make check-cxx # Test
  $ make install-cxx install-cxxabi # Install

For more information about configuring libc++ see :ref:`CMake Options`. You may
also want to read the `LLVM getting started
<https://llvm.org/docs/GettingStarted.html>`_ documentation.

Shared libraries for libc++ and libc++ abi should now be present in
``build/lib``.  See :ref:`using an alternate libc++ installation <alternate
libcxx>` for information on how to use this libc++.

The instructions are for building libc++ on
FreeBSD, Linux, or Mac using `libc++abi`_ as the C++ ABI library.
On Linux, it is also possible to use :ref:`libsupc++ <libsupcxx>` or libcxxrt.

It is possible to build libc++ standalone (i.e. without building other LLVM
projects). A standalone build would look like this:

.. code-block:: bash

  $ git clone https://github.com/llvm/llvm-project.git llvm-project
  $ cd llvm-project
  $ mkdir build && cd build
  $ cmake -DCMAKE_C_COMPILER=clang \
          -DCMAKE_CXX_COMPILER=clang++ \
          -DLIBCXX_CXX_ABI=libcxxabi \
          -DLIBCXX_CXX_ABI_INCLUDE_PATHS=path/to/separate/libcxxabi/include \
          ../libcxx
  $ make
  $ make check-cxx # optional


Support for Windows
-------------------

libcxx supports being built with clang-cl, but not with MSVC's cl.exe, as
cl doesn't support the ``#include_next`` extension. Furthermore, VS 2017 or
newer (19.14) is required.

libcxx also supports being built with clang targeting MinGW environments.

CMake + Visual Studio
~~~~~~~~~~~~~~~~~~~~~

Building with Visual Studio currently does not permit running tests. However,
it is the simplest way to build.

.. code-block:: batch

  > cmake -G "Visual Studio 16 2019"              ^
          -T "ClangCL"                            ^
          -DLIBCXX_ENABLE_SHARED=YES              ^
          -DLIBCXX_ENABLE_STATIC=NO               ^
          -DLIBCXX_ENABLE_EXPERIMENTAL_LIBRARY=NO ^
          \path\to\libcxx
  > cmake --build .

CMake + ninja (MSVC)
~~~~~~~~~~~~~~~~~~~~

Building with ninja is required for development to enable tests.
A couple of tests require Bash to be available, and a couple dozens
of tests require other posix tools (cp, grep and similar - LLVM's tests
require the same). Without those tools the vast majority of tests
can still be ran successfully.

If Git for Windows is available, that can be used to provide the bash
shell by adding the right bin directory to the path, e.g.
``set PATH=%PATH%;C:\Program Files\Git\usr\bin``.

Alternatively, one can also choose to run the whole build in a MSYS2
shell. That can be set up e.g. by starting a Visual Studio Tools Command
Prompt (for getting the environment variables pointing to the headers and
import libraries), and making sure that clang-cl is available in the
path. From there, launch an MSYS2 shell via e.g.
``C:\msys64\msys2_shell.cmd -full-path -mingw64`` (preserving the earlier
environment, allowing the MSVC headers/libraries and clang-cl to be found).

In either case, then run:

.. code-block:: batch

  > cmake -G Ninja                                                                    ^
          -DCMAKE_BUILD_TYPE=Release                                                  ^
          -DCMAKE_C_COMPILER=clang-cl                                                 ^
          -DCMAKE_CXX_COMPILER=clang-cl                                               ^
          -DLIBCXX_ENABLE_EXPERIMENTAL_LIBRARY=NO                                     ^
          path/to/libcxx
  > ninja cxx
  > ninja check-cxx

If you are running in an MSYS2 shell and you have installed the
MSYS2-provided clang package (which defaults to a non-MSVC target), you
should add e.g. ``-DLIBCXX_TARGET_TRIPLE=x86_64-windows-msvc`` (replacing
``x86_64`` with the architecture you're targeting) to the ``cmake`` command
line above. This will instruct ``check-cxx`` to use the right target triple
when invoking ``clang++``.

Also note that if not building in Release mode, a failed assert in the tests
pops up a blocking dialog box, making it hard to run a larger number of tests.

CMake + ninja (MinGW)
~~~~~~~~~~~~~~~~~~~~~

libcxx can also be built in MinGW environments, e.g. with the MinGW
compilers in MSYS2. This requires clang to be available (installed with
e.g. the ``mingw-w64-x86_64-clang`` package), together with CMake and ninja.

.. code-block:: bash

  > cmake -G Ninja                                                                    \
          -DCMAKE_C_COMPILER=clang                                                    \
          -DCMAKE_CXX_COMPILER=clang++                                                \
          -DLIBCXX_HAS_WIN32_THREAD_API=ON                                            \
          -DLIBCXX_CXX_ABI=libstdc++                                                  \
          -DLIBCXX_TARGET_INFO="libcxx.test.target_info.MingwLocalTI"                 \
          path/to/libcxx
  > ninja cxx
  > cp /mingw64/bin/{libstdc++-6,libgcc_s_seh-1,libwinpthread-1}.dll lib
  > ninja check-cxx

As this build configuration ends up depending on a couple other DLLs that
aren't available in path while running tests, copy them into the same
directory as the tested libc++ DLL.

(Building a libc++ that depends on libstdc++ isn't necessarily a config one
would want to deploy, but it simplifies the config for testing purposes.)

.. _`libc++abi`: http://libcxxabi.llvm.org/


.. _CMake Options:

CMake Options
=============

Here are some of the CMake variables that are used often, along with a
brief explanation and LLVM-specific notes. For full documentation, check the
CMake docs or execute ``cmake --help-variable VARIABLE_NAME``.

**CMAKE_BUILD_TYPE**:STRING
  Sets the build type for ``make`` based generators. Possible values are
  Release, Debug, RelWithDebInfo and MinSizeRel. On systems like Visual Studio
  the user sets the build type with the IDE settings.

**CMAKE_INSTALL_PREFIX**:PATH
  Path where LLVM will be installed if "make install" is invoked or the
  "INSTALL" target is built.

**CMAKE_CXX_COMPILER**:STRING
  The C++ compiler to use when building and testing libc++.


.. _libcxx-specific options:

libc++ specific options
-----------------------

.. option:: LIBCXX_INSTALL_LIBRARY:BOOL

  **Default**: ``ON``

  Toggle the installation of the library portion of libc++.

.. option:: LIBCXX_INSTALL_HEADERS:BOOL

  **Default**: ``ON``

  Toggle the installation of the libc++ headers.

.. option:: LIBCXX_ENABLE_ASSERTIONS:BOOL

  **Default**: ``OFF``

  Build libc++ with assertions enabled.

.. option:: LIBCXX_BUILD_32_BITS:BOOL

  **Default**: ``OFF``

  Build libc++ as a 32 bit library. Also see `LLVM_BUILD_32_BITS`.

.. option:: LIBCXX_ENABLE_SHARED:BOOL

  **Default**: ``ON``

  Build libc++ as a shared library. Either `LIBCXX_ENABLE_SHARED` or
  `LIBCXX_ENABLE_STATIC` has to be enabled.

.. option:: LIBCXX_ENABLE_STATIC:BOOL

  **Default**: ``ON``

  Build libc++ as a static library. Either `LIBCXX_ENABLE_SHARED` or
  `LIBCXX_ENABLE_STATIC` has to be enabled.

.. option:: LIBCXX_LIBDIR_SUFFIX:STRING

  Extra suffix to append to the directory where libraries are to be installed.
  This option overrides `LLVM_LIBDIR_SUFFIX`.

.. option:: LIBCXX_HERMETIC_STATIC_LIBRARY:BOOL

  **Default**: ``OFF``

  Do not export any symbols from the static libc++ library.
  This is useful when the static libc++ library is being linked into shared
  libraries that may be used in with other shared libraries that use different
  C++ library. We want to avoid exporting any libc++ symbols in that case.

.. option:: LIBCXX_ENABLE_FILESYSTEM:BOOL

   **Default**: ``ON`` except on Windows.

   This option can be used to enable or disable the filesystem components on
   platforms that may not support them. For example on Windows.

.. _libc++experimental options:

libc++experimental Specific Options
------------------------------------

.. option:: LIBCXX_ENABLE_EXPERIMENTAL_LIBRARY:BOOL

  **Default**: ``ON``

  Build and test libc++experimental.a.

.. option:: LIBCXX_INSTALL_EXPERIMENTAL_LIBRARY:BOOL

  **Default**: ``LIBCXX_ENABLE_EXPERIMENTAL_LIBRARY AND LIBCXX_INSTALL_LIBRARY``

  Install libc++experimental.a alongside libc++.


.. _ABI Library Specific Options:

ABI Library Specific Options
----------------------------

.. option:: LIBCXX_CXX_ABI:STRING

  **Values**: ``none``, ``libcxxabi``, ``libcxxrt``, ``libstdc++``, ``libsupc++``.

  Select the ABI library to build libc++ against.

.. option:: LIBCXX_CXX_ABI_INCLUDE_PATHS:PATHS

  Provide additional search paths for the ABI library headers.

.. option:: LIBCXX_CXX_ABI_LIBRARY_PATH:PATH

  Provide the path to the ABI library that libc++ should link against.

.. option:: LIBCXX_ENABLE_STATIC_ABI_LIBRARY:BOOL

  **Default**: ``OFF``

  If this option is enabled, libc++ will try and link the selected ABI library
  statically.

.. option:: LIBCXX_ENABLE_ABI_LINKER_SCRIPT:BOOL

  **Default**: ``ON`` by default on UNIX platforms other than Apple unless
  'LIBCXX_ENABLE_STATIC_ABI_LIBRARY' is ON. Otherwise the default value is ``OFF``.

  This option generate and installs a linker script as ``libc++.so`` which
  links the correct ABI library.

.. option:: LIBCXXABI_USE_LLVM_UNWINDER:BOOL

  **Default**: ``OFF``

  Build and use the LLVM unwinder. Note: This option can only be used when
  libc++abi is the C++ ABI library used.


libc++ Feature Options
----------------------

.. option:: LIBCXX_ENABLE_EXCEPTIONS:BOOL

  **Default**: ``ON``

  Build libc++ with exception support.

.. option:: LIBCXX_ENABLE_RTTI:BOOL

  **Default**: ``ON``

  Build libc++ with run time type information.

.. option:: LIBCXX_INCLUDE_TESTS:BOOL

  **Default**: ``ON`` (or value of ``LLVM_INCLUDE_TESTS``)

  Build the libc++ tests.

.. option:: LIBCXX_INCLUDE_BENCHMARKS:BOOL

  **Default**: ``ON``

  Build the libc++ benchmark tests and the Google Benchmark library needed
  to support them.

.. option:: LIBCXX_BENCHMARK_TEST_ARGS:STRING

  **Default**: ``--benchmark_min_time=0.01``

  A semicolon list of arguments to pass when running the libc++ benchmarks using the
  ``check-cxx-benchmarks`` rule. By default we run the benchmarks for a very short amount of time,
  since the primary use of ``check-cxx-benchmarks`` is to get test and sanitizer coverage, not to
  get accurate measurements.

.. option:: LIBCXX_BENCHMARK_NATIVE_STDLIB:STRING

  **Default**:: ``""``

  **Values**:: ``libc++``, ``libstdc++``

  Build the libc++ benchmark tests and Google Benchmark library against the
  specified standard library on the platform. On Linux this can be used to
  compare libc++ to libstdc++ by building the benchmark tests against both
  standard libraries.

.. option:: LIBCXX_BENCHMARK_NATIVE_GCC_TOOLCHAIN:STRING

  Use the specified GCC toolchain and standard library when building the native
  stdlib benchmark tests.

.. option:: LIBCXX_HIDE_FROM_ABI_PER_TU_BY_DEFAULT:BOOL

  **Default**: ``OFF``

  Pick the default for whether to constrain ABI-unstable symbols to
  each individual translation unit. This setting controls whether
  `_LIBCPP_HIDE_FROM_ABI_PER_TU_BY_DEFAULT` is defined by default --
  see the documentation of that macro for details.


libc++ ABI Feature Options
--------------------------

The following options allow building libc++ for a different ABI version.

.. option:: LIBCXX_ABI_VERSION:STRING

  **Default**: ``1``

  Defines the target ABI version of libc++.

.. option:: LIBCXX_ABI_UNSTABLE:BOOL

  **Default**: ``OFF``

  Build the "unstable" ABI version of libc++. Includes all ABI changing features
  on top of the current stable version.

.. option:: LIBCXX_ABI_NAMESPACE:STRING

  **Default**: ``__n`` where ``n`` is the current ABI version.

  This option defines the name of the inline ABI versioning namespace. It can be used for building
  custom versions of libc++ with unique symbol names in order to prevent conflicts or ODR issues
  with other libc++ versions.

  .. warning::
    When providing a custom namespace, it's the users responsibility to ensure the name won't cause
    conflicts with other names defined by libc++, both now and in the future. In particular, inline
    namespaces of the form ``__[0-9]+`` are strictly reserved by libc++ and may not be used by users.
    Doing otherwise could cause conflicts and hinder libc++ ABI evolution.

.. option:: LIBCXX_ABI_DEFINES:STRING

  **Default**: ``""``

  A semicolon-separated list of ABI macros to persist in the site config header.
  See ``include/__config`` for the list of ABI macros.


.. _LLVM-specific variables:

LLVM-specific options
---------------------

.. option:: LLVM_LIBDIR_SUFFIX:STRING

  Extra suffix to append to the directory where libraries are to be
  installed. On a 64-bit architecture, one could use ``-DLLVM_LIBDIR_SUFFIX=64``
  to install libraries to ``/usr/lib64``.

.. option:: LLVM_BUILD_32_BITS:BOOL

  Build 32-bits executables and libraries on 64-bits systems. This option is
  available only on some 64-bits Unix systems. Defaults to OFF.

.. option:: LLVM_LIT_ARGS:STRING

  Arguments given to lit.  ``make check`` and ``make clang-test`` are affected.
  By default, ``'-sv --no-progress-bar'`` on Visual C++ and Xcode, ``'-sv'`` on
  others.


Using Alternate ABI libraries
=============================


.. _libsupcxx:

Using libsupc++ on Linux
------------------------

You will need libstdc++ in order to provide libsupc++.

Figure out where the libsupc++ headers are on your system. On Ubuntu this
is ``/usr/include/c++/<version>`` and ``/usr/include/c++/<version>/<target-triple>``

You can also figure this out by running

.. code-block:: bash

  $ echo | g++ -Wp,-v -x c++ - -fsyntax-only
  ignoring nonexistent directory "/usr/local/include/x86_64-linux-gnu"
  ignoring nonexistent directory "/usr/lib/gcc/x86_64-linux-gnu/4.7/../../../../x86_64-linux-gnu/include"
  #include "..." search starts here:
  #include &lt;...&gt; search starts here:
  /usr/include/c++/4.7
  /usr/include/c++/4.7/x86_64-linux-gnu
  /usr/include/c++/4.7/backward
  /usr/lib/gcc/x86_64-linux-gnu/4.7/include
  /usr/local/include
  /usr/lib/gcc/x86_64-linux-gnu/4.7/include-fixed
  /usr/include/x86_64-linux-gnu
  /usr/include
  End of search list.

Note that the first two entries happen to be what we are looking for. This
may not be correct on other platforms.

We can now run CMake:

.. code-block:: bash

  $ CC=clang CXX=clang++ cmake -G "Unix Makefiles" \
    -DLIBCXX_CXX_ABI=libstdc++ \
    -DLIBCXX_CXX_ABI_INCLUDE_PATHS="/usr/include/c++/4.7/;/usr/include/c++/4.7/x86_64-linux-gnu/" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr \
    <libc++-source-dir>


You can also substitute ``-DLIBCXX_CXX_ABI=libsupc++``
above, which will cause the library to be linked to libsupc++ instead
of libstdc++, but this is only recommended if you know that you will
never need to link against libstdc++ in the same executable as libc++.
GCC ships libsupc++ separately but only as a static library.  If a
program also needs to link against libstdc++, it will provide its
own copy of libsupc++ and this can lead to subtle problems.

.. code-block:: bash

  $ make cxx
  $ make install

You can now run clang with -stdlib=libc++.


.. _libcxxrt_ref:

Using libcxxrt on Linux
------------------------

You will need to keep the source tree of `libcxxrt`_ available
on your build machine and your copy of the libcxxrt shared library must
be placed where your linker will find it.

We can now run CMake like:

.. code-block:: bash

  $ CC=clang CXX=clang++ cmake -G "Unix Makefiles" \
          -DLIBCXX_CXX_ABI=libcxxrt \
          -DLIBCXX_CXX_ABI_INCLUDE_PATHS=path/to/libcxxrt-sources/src \
                -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_INSTALL_PREFIX=/usr \
                <libc++-source-directory>
  $ make cxx
  $ make install

Unfortunately you can't simply run clang with "-stdlib=libc++" at this point, as
clang is set up to link for libc++ linked to libsupc++.  To get around this
you'll have to set up your linker yourself (or patch clang).  For example,

.. code-block:: bash

  $ clang++ -stdlib=libc++ helloworld.cpp \
            -nodefaultlibs -lc++ -lcxxrt -lm -lc -lgcc_s -lgcc

Alternately, you could just add libcxxrt to your libraries list, which in most
situations will give the same result:

.. code-block:: bash

  $ clang++ -stdlib=libc++ helloworld.cpp -lcxxrt

.. _`libcxxrt`: https://github.com/pathscale/libcxxrt/


Using a local ABI library installation
---------------------------------------

.. warning::
  This is not recommended in almost all cases.

These instructions should only be used when you can't install your ABI library.

Normally you must link libc++ against a ABI shared library that the
linker can find.  If you want to build and test libc++ against an ABI
library not in the linker's path you need to set
``-DLIBCXX_CXX_ABI_LIBRARY_PATH=/path/to/abi/lib`` when configuring CMake.

An example build using libc++abi would look like:

.. code-block:: bash

  $ CC=clang CXX=clang++ cmake \
              -DLIBCXX_CXX_ABI=libc++abi  \
              -DLIBCXX_CXX_ABI_INCLUDE_PATHS="/path/to/libcxxabi/include" \
              -DLIBCXX_CXX_ABI_LIBRARY_PATH="/path/to/libcxxabi-build/lib" \
               path/to/libcxx
  $ make

When testing libc++ LIT will automatically link against the proper ABI
library.
