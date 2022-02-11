.. _BuildingLibcxx:

===============
Building libc++
===============

.. contents::
  :local:

.. _build instructions:

The instructions on this page are aimed at vendors who ship libc++ as part of an
operating system distribution, a toolchain or similar shipping vehicules. If you
are a user merely trying to use libc++ in your program, you most likely want to
refer to your vendor's documentation, or to the general documentation for using
libc++ :ref:`here <using-libcxx>`.

.. warning::
  If your operating system already provides libc++, it is important to be careful
  not to replace it. Replacing your system's libc++ installation could render it
  non-functional. Use the CMake option ``CMAKE_INSTALL_PREFIX`` to select a safe
  place to install libc++.


The default build
=================

The default way of building libc++, libc++abi and libunwind is to root the CMake
invocation at ``<monorepo>/runtimes``. While those projects are under the LLVM
umbrella, they are different in nature from other build tools, so it makes sense
to treat them as a separate set of entities. The default build can be achieved
with the following CMake invocation:

.. code-block:: bash

  $ git clone https://github.com/llvm/llvm-project.git
  $ cd llvm-project
  $ mkdir build
  $ cmake -G Ninja -S runtimes -B build -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;libunwind" # Configure
  $ ninja -C build cxx cxxabi unwind                                                        # Build
  $ ninja -C build check-cxx check-cxxabi check-unwind                                      # Test
  $ ninja -C build install-cxx install-cxxabi install-unwind                                # Install

.. note::
  See :ref:`CMake Options` below for more configuration options.

After building the various ``install-XXX`` targets, shared libraries for libc++, libc++abi and
libunwind should now be present in ``<CMAKE_INSTALL_PREFIX>/lib``, and headers in
``<CMAKE_INSTALL_PREFIX>/include/c++/v1``. See :ref:`using an alternate libc++ installation
<alternate libcxx>` for information on how to use this libc++ over the default one.

In the default configuration, the runtimes will be built using the compiler available by default
on your system. Of course, you can change what compiler is being used with the usual CMake
variables. If you wish to build the runtimes from a just-built Clang, the bootstrapping build
explained below makes this task easy.


Bootstrapping build
===================

It is possible to build Clang and then build the runtimes using that just-built compiler in a
single CMake invocation. This is usually the correct way to build the runtimes when putting together
a toolchain, or when the system compiler is not adequate to build them (too old, unsupported, etc.).
To do this, use the following CMake invocation, and in particular notice how we're now rooting the
CMake invocation at ``<monorepo>/llvm``:

.. code-block:: bash

  $ mkdir build
  $ cmake -G Ninja -S llvm -B build -DLLVM_ENABLE_PROJECTS="clang"                      \  # Configure
                                    -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;libunwind" \
                                    -DLLVM_RUNTIME_TARGETS="<target-triple>"
  $ ninja -C build runtimes                                                                # Build
  $ ninja -C build check-runtimes                                                          # Test
  $ ninja -C build install-runtimes                                                        # Install

.. note::
  This type of build is also commonly called a "Runtimes build", but we would like to move
  away from that terminology, which is too confusing.

Support for Windows
===================

libcxx supports being built with clang-cl, but not with MSVC's cl.exe, as
cl doesn't support the ``#include_next`` extension. Furthermore, VS 2017 or
newer (19.14) is required.

libcxx also supports being built with clang targeting MinGW environments.

CMake + Visual Studio
---------------------

Building with Visual Studio currently does not permit running tests. However,
it is the simplest way to build.

.. code-block:: batch

  > cmake -G "Visual Studio 16 2019" -S libcxx -B build ^
          -T "ClangCL"                                  ^
          -DLIBCXX_ENABLE_SHARED=YES                    ^
          -DLIBCXX_ENABLE_STATIC=NO                     ^
          -DLIBCXX_ENABLE_EXPERIMENTAL_LIBRARY=NO
  > cmake --build build

CMake + ninja (MSVC)
--------------------

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

  > cmake -G Ninja -S libcxx -B build                                                 ^
          -DCMAKE_C_COMPILER=clang-cl                                                 ^
          -DCMAKE_CXX_COMPILER=clang-cl                                               ^
          -DLIBCXX_ENABLE_EXPERIMENTAL_LIBRARY=NO
  > ninja -C build cxx
  > ninja -C build check-cxx

If you are running in an MSYS2 shell and you have installed the
MSYS2-provided clang package (which defaults to a non-MSVC target), you
should add e.g. ``-DLIBCXX_TARGET_TRIPLE=x86_64-windows-msvc`` (replacing
``x86_64`` with the architecture you're targeting) to the ``cmake`` command
line above. This will instruct ``check-cxx`` to use the right target triple
when invoking ``clang++``.

Also note that if not building in Release mode, a failed assert in the tests
pops up a blocking dialog box, making it hard to run a larger number of tests.

CMake + ninja (MinGW)
---------------------

libcxx can also be built in MinGW environments, e.g. with the MinGW
compilers in MSYS2. This requires clang to be available (installed with
e.g. the ``mingw-w64-x86_64-clang`` package), together with CMake and ninja.

.. code-block:: bash

  > cmake -G Ninja -S libcxx -B build                                                 \
          -DCMAKE_C_COMPILER=clang                                                    \
          -DCMAKE_CXX_COMPILER=clang++                                                \
          -DLIBCXX_HAS_WIN32_THREAD_API=ON                                            \
          -DLIBCXX_CXX_ABI=libstdc++                                                  \
          -DLIBCXX_TARGET_INFO="libcxx.test.target_info.MingwLocalTI"
  > ninja -C build cxx
  > cp /mingw64/bin/{libstdc++-6,libgcc_s_seh-1,libwinpthread-1}.dll lib
  > ninja -C build check-cxx

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

   **Default**: ``ON`` except on Windows when using MSVC.

   This option can be used to enable or disable the filesystem components on
   platforms that may not support them. For example on Windows when using MSVC.

.. option:: LIBCXX_ENABLE_WIDE_CHARACTERS:BOOL

   **Default**: ``ON``

   This option can be used to disable support for ``wchar_t`` in the library. It also
   allows the library to work on top of a C Standard Library that does not provide
   support for ``wchar_t``. This is especially useful in embedded settings where
   C Standard Libraries don't always provide all the usual bells and whistles.

.. option:: LIBCXX_ENABLE_INCOMPLETE_FEATURES:BOOL

  **Default**: ``ON``

  Whether to enable support for incomplete library features. Incomplete features
  are new library features under development. These features don't guarantee
  ABI stability nor the quality of completed library features. Vendors
  shipping the library may want to disable this option.

.. option:: LIBCXX_INSTALL_LIBRARY_DIR:PATH

  **Default**: ``lib${LIBCXX_LIBDIR_SUFFIX}``

  Path where built libc++ libraries should be installed. If a relative path,
  relative to ``CMAKE_INSTALL_PREFIX``.

.. option:: LIBCXX_INSTALL_INCLUDE_DIR:PATH

  **Default**: ``include/c++/v1``

  Path where target-agnostic libc++ headers should be installed. If a relative
  path, relative to ``CMAKE_INSTALL_PREFIX``.

.. option:: LIBCXX_INSTALL_INCLUDE_TARGET_DIR:PATH

  **Default**: ``include/c++/v1`` or
  ``include/${LLVM_DEFAULT_TARGET_TRIPLE}/c++/v1``

  Path where target-specific libc++ headers should be installed. If a relative
  path, relative to ``CMAKE_INSTALL_PREFIX``.

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
    When providing a custom namespace, it's the user's responsibility to ensure the name won't cause
    conflicts with other names defined by libc++, both now and in the future. In particular, inline
    namespaces of the form ``__[0-9]+`` could cause conflicts with future versions of the library,
    and so should be avoided.

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

In order to implement various features like exceptions, RTTI, ``dynamic_cast`` and
more, libc++ requires what we refer to as an ABI library. Typically, that library
implements the `Itanium C++ ABI <https://itanium-cxx-abi.github.io/cxx-abi/abi.html>`_.

By default, libc++ uses libc++abi as an ABI library. However, it is possible to use
other ABI libraries too.

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
may not be correct on all platforms.

We can now run CMake:

.. code-block:: bash

  $ cmake -G Ninja -S runtimes -B build       \
    -DLLVM_ENABLE_RUNTIMES="libcxx"           \
    -DLIBCXX_CXX_ABI=libstdc++                \
    -DLIBCXX_CXX_ABI_INCLUDE_PATHS="/usr/include/c++/4.7/;/usr/include/c++/4.7/x86_64-linux-gnu/"
  $ ninja -C build install-cxx


You can also substitute ``-DLIBCXX_CXX_ABI=libsupc++``
above, which will cause the library to be linked to libsupc++ instead
of libstdc++, but this is only recommended if you know that you will
never need to link against libstdc++ in the same executable as libc++.
GCC ships libsupc++ separately but only as a static library.  If a
program also needs to link against libstdc++, it will provide its
own copy of libsupc++ and this can lead to subtle problems.

Using libcxxrt on Linux
------------------------

You will need to keep the source tree of `libcxxrt`_ available
on your build machine and your copy of the libcxxrt shared library must
be placed where your linker will find it.

We can now run CMake like:

.. code-block:: bash

  $ cmake -G Ninja -S runtimes -B build                               \
          -DLLVM_ENABLE_RUNTIMES="libcxx"                             \
          -DLIBCXX_CXX_ABI=libcxxrt                                   \
          -DLIBCXX_CXX_ABI_INCLUDE_PATHS=path/to/libcxxrt-sources/src
  $ ninja -C build install-cxx

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

.. _`libcxxrt`: https://github.com/libcxxrt/libcxxrt
