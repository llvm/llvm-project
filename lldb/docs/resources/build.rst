Build
=====

.. contents::
   :local:

Building LLDB on Windows
------------------------

**Required Dependencies**

* Visual Studio 2015 or greater
* Windows SDK 8.0 or higher. In general it is best to use the latest available version.
* `Python 3.5 or higher <https://www.python.org/downloads/windows/>`_ or
  higher. Earlier versions of Python can be made to work by compiling your own
  distribution from source, but this workflow is unsupported and you are own
  your own.
* `Ninja build tool <https://ninja-build.org/>`_ (strongly recommended)
* `GnuWin32 <http://gnuwin32.sourceforge.net/>`_
* `SWIG for Windows <http://www.swig.org/download.html>`_ (version 3+)


**Optional Dependencies**

* `Python Tools for Visual Studio
  <https://github.com/Microsoft/PTVS/releases>`_. If you plan to debug test
  failures or even write new tests at all, PTVS is an indispensable debugging
  extension to VS that enables full editing and debugging support for Python
  (including mixed native/managed debugging)

**Preliminaries**

This section describes how to set up your system and install the required
dependencies such that they can be found when needed during the build process.
The steps outlined here only need to be performed once.

#. Install Visual Studio and the Windows SDK.
#. Install GnuWin32, making sure ``<GnuWin32 install dir>\bin`` is added to
   your PATH environment variable.
#. Install SWIG for Windows, making sure ``<SWIG install dir>`` is added to
   your PATH environment variable.

**Building LLDB**

Any command prompt from which you build LLDB should have a valid Visual Studio
environment setup. This means you should run ``vcvarsall.bat`` or open an
appropriate Visual Studio Command Prompt corresponding to the version you wish
to use.


Finally, when you are ready to build LLDB, generate CMake with the following
command line:

::

  cmake -G Ninja <cmake variables> <path to root of llvm src tree>


and run ``ninja`` to build LLDB. Information about running the LLDB test suite
can be found on the test page.


Following is a description of some of the most important CMake variables which
you are likely to encounter. A variable FOO is set by adding ``-DFOO=value`` to
the CMake command line.

* ``LLDB_TEST_DEBUG_TEST_CRASHES`` (Default=0): If set to 1, will cause Windows
  to generate a crash dialog whenever lldb.exe or the python extension module
  crashes while running the test suite. If set to 0, LLDB will silently crash.
  Setting to 1 allows a developer to attach a JIT debugger at the time of a
  crash, rather than having to reproduce a failure or use a crash dump.
* ``PYTHON_HOME`` (Required): Path to the folder where the Python distribution
  is installed. For example, ``C:\Python35``.
* ``LLDB_RELOCATABLE_PYTHON`` (Default=0): When this is 0, LLDB will bind
  statically to the location specified in the ``PYTHON_HOME`` CMake variable,
  ignoring any value of ``PYTHONHOME`` set in the environment. This is most
  useful for developers who simply want to run LLDB after they build it. If you
  wish to move a build of LLDB to a different machine where Python will be in a
  different location, setting ``LLDB_RELOCATABLE_PYTHON`` to 1 will cause
  Python to use its default mechanism for finding the python installation at
  runtime (looking for installed Pythons, or using the ``PYTHONHOME``
  environment variable if it is specified).
* ``LLDB_TEST_C_COMPILER`` or ``LLDB_TEST_CXX_COMPILER``: The test suite needs
  to be able to find a copy of clang.exe that it can use to compile inferior
  programs. Note that MSVC is not supported here, it must be a path to a clang
  executable. Note that using a release clang.exe is strongly recommended here,
  as it will make the test suite run much faster. This can be a path to any
  recent clang.exe, including one you built yourself. These variables are
  ignored unless the respective ``LLDB_TEST_USE_CUSTOM_C_COMPILER`` and
  ``LLDB_TEST_USE_CUSTOM_CXX_COMPILER`` are set to ON.

Sample command line:

::

  cmake -G Ninja -DLLDB_TEST_DEBUG_TEST_CRASHES=1 -DPYTHON_HOME=C:\Python35 -DLLDB_TEST_USE_CUSTOM_C_COMPILER=ON -DLLDB_TEST_C_COMPILER=d:\src\llvmbuild\ninja_release\bin\clang.exe ..\..\llvm


**Working with both Ninja and MSVC**

Compiling with ninja is both faster and simpler than compiling with MSVC, but
chances are you still want to debug LLDB with MSVC (at least until we can debug
LLDB on Windows with LLDB!). One solution to this is to run cmake twice and
generate the output into two different folders. One for compiling (the ninja
folder), and one for editing / browsing / debugging (the MSVC folder).

To do this, simply run ``cmake -G Ninja <arguments>`` from one folder, and
``cmake -G "Visual Studio 14 2015" <arguments>`` in another folder. Then you
can open the .sln file in Visual Studio, set lldb as the startup project, and
use F5 to run it. You need only edit the project settings to set the executable
and the working directory to point to binaries inside of the ninja tree.

Building LLDB on macOS
----------------------

There are two ways to build LLDB on macOS: Using Xcode and using CMake

**Preliminaries**

In addition to any dependencies required by LLVM and Clang, LLDB needs a few
development packages that may also need to be installed depending on your
system. The current list of dependencies are:

* Xcode 4.3 or newer requires the "Command Line Tools" component (XCode->Preferences->Downloads->Components)
* `Swig <http://swig.org/>`_

**Building LLDB with Xcode**

Building on macOS with Xcode is as easy as downloading the code and building
the Xcode project or workspace:

* Download the lldb sources.
* Follow the code signing instructions in ``lldb/docs/code-signing.txt``.
* In Xcode, open ``lldb/lldb.xcworkspace``, select the lldb-tool scheme, and build.

**Building LLDB with CMake**

First download the LLVM, Clang, libc++ and LLDB sources. Refer to this page for
precise instructions on this step.

Refer to the code signing instructions in ``lldb/docs/code-signing.txt`` for
info on codesigning debugserver during the build.

Using CMake is documented on the `Building LLVM with CMake
<http://llvm.org/docs/CMake.html>`_ page. Ninja is the recommended generator to
use when building LLDB with CMake.

::

  > cmake $PATH_TO_LLVM -G Ninja
  > ninja lldb

As noted in the "Building LLVM with CMake" page mentioned above, you can pass
variables to cmake to change build behavior. If LLDB is built as a part of
LLVM, then you can pass LLVM-specific CMake variables to cmake when building
LLDB.

If you are building both Clang and LLDB together, be sure to also add libc++,
which is currently required for testing on macOS:

::

  > cmake -D LLVM_ENABLE_PROJECTS='clang;lldb;libcxx' $PATH_TO_LLVM -G Ninja

Here are some commonly used LLDB-specific CMake variables:

* ``LLDB_EXPORT_ALL_SYMBOLS:BOOL`` : Exports all symbols. Useful in conjunction
  with CMAKE_BUILD_TYPE=Debug.
* ``LLDB_BUILD_FRAMEWORK:BOOL`` : Builds LLDB.framework as Xcode would
* ``LLDB_CODESIGN_IDENTITY:STRING`` : Determines the codesign identity to use.
  An empty string means skip building debugserver to avoid codesigning.

Building LLDB on Linux, FreeBSD and NetBSD
------------------------------------------

This document describes the steps needed to compile LLDB on most Linux systems,
FreeBSD and NetBSD.

**Preliminaries**

LLDB relies on many of the technologies developed by the larger LLVM project.
In particular, it requires both Clang and LLVM itself in order to build. Due to
this tight integration the Getting Started guides for both of these projects
come as prerequisite reading:

* `LLVM <http://llvm.org/docs/GettingStarted.html>`_
* `Clang <http://clang.llvm.org/get_started.html>`_

Supported compilers for building LLDB on Linux include:

* Clang 3.2
* GCC 4.6.2 (later versions should work as well)

It is recommended to use libstdc++ 4.6 (or higher) to build LLDB on Linux, but
using libc++ is also known to work.

On FreeBSD the base system Clang and libc++ may be used to build LLDB, or the
GCC port or package.

On NetBSD the base system GCC and libstdc++ are used to build LLDB, Clang/LLVM
and libc++ should also work.

In addition to any dependencies required by LLVM and Clang, LLDB needs a few
development packages that may also need to be installed depending on your
system. The current list of dependencies are:

* `Swig <http://swig.org/>`_
* `libedit (Linux only) <http://www.thrysoee.dk/editline>`_
* `Python <http://www.python.org/>`_


So for example, on a Fedora system one might run:

::

  > yum install libedit-devel libxml2-devel ncurses-devel python-devel swig

On a Debian or Ubuntu system one might run:

::

  > sudo apt-get install build-essential subversion swig python2.7-dev libedit-dev libncurses5-dev

or

::

  > sudo apt-get build-dep lldb-3.3 # or lldb-3.4


On FreeBSD one might run:

::

  > pkg install swig python

On NetBSD one might run:

::

  > pkgin install swig python27 cmake ninja-build

**To build with CMake**

Using CMake is documented on the `Building LLVM with CMake
<http://llvm.org/docs/CMake.html>`_ page. Building LLDB is possible using one
of the following generators:

* Ninja
* Unix Makefiles

**Using CMake + Ninja**

Ninja is the fastest way to build LLDB! In order to use ninja, you need to have
recent versions of CMake and ninja on your system. To build using ninja:

::

  > cmake ../llvm -G Ninja
  > ninja lldb
  > ninja check-lldb


If you want to debug the lldb that you're building -- that is, build it with
debug info enabled -- pass two additional arguments to cmake before running
ninja:

::

  > cmake ../llvm -G Ninja -DLLDB_EXPORT_ALL_SYMBOLS=1 -DCMAKE_BUILD_TYPE=Debug


**Using CMake + Unix Makefiles**

If you do not have Ninja, you can still use CMake to generate Unix Makefiles that build LLDB:

::

  > cmake ..
  > make
  > make check-lldb


**Building API reference documentation**

LLDB exposes a C++ as well as a Python API. To build the reference
documentation for these two APIs, ensure you have the required dependencies
installed, and build the ``lldb-python-doc`` and ``lldb-cpp-doc`` CMake
targets.

The output HTML reference documentation can be found in
``<build-dir>/tools/lldb/docs/``.

**Additional Notes**

LLDB has a Python scripting capability and supplies its own Python module named
lldb. If a script is run inside the command line lldb application, the Python
module is made available automatically. However, if a script is to be run by a
Python interpreter outside the command line application, the ``PYTHONPATH``
environment variable can be used to let the Python interpreter find the lldb
module.

Current stable NetBSD release doesn't ship with libpanel(3), therefore it's
required to disable curses(3) support with the
``-DLLDB_DISABLE_CURSES:BOOL=TRUE`` option. To make sure check if
``/usr/include/panel.h`` exists in your system.

The correct path can be obtained by invoking the command line lldb tool with
the -P flag:

::

  > export PYTHONPATH=`$llvm/build/Debug+Asserts/bin/lldb -P`

If you used a different build directory or made a release build, you may need
to adjust the above to suit your needs. To test that the lldb Python module is
built correctly and is available to the default Python interpreter, run:

::

  > python -c 'import lldb'

**Cross-compiling LLDB**


In order to debug remote targets running different architectures than your
host, you will need to compile LLDB (or at least the server component) for the
target. While the easiest solution is to just compile it locally on the target,
this is often not feasible, and in these cases you will need to cross-compile
LLDB on your host.

Cross-compilation is often a daunting task and has a lot of quirks which depend
on the exact host and target architectures, so it is not possible to give a
universal guide which will work on all platforms. However, here we try to
provide an overview of the cross-compilation process along with the main things
you should look out for.

First, you will need a working toolchain which is capable of producing binaries
for the target architecture. Since you already have a checkout of clang and
lldb, you can compile a host version of clang in a separate folder and use
that. Alternatively you can use system clang or even cross-gcc if your
distribution provides such packages (e.g., ``g++-aarch64-linux-gnu`` on
Ubuntu).

Next, you will need a copy of the required target headers and libraries on your
host. The libraries can be usually obtained by copying from the target machine,
however the headers are often not found there, especially in case of embedded
platforms. In this case, you will need to obtain them from another source,
either a cross-package if one is available, or cross-compiling the respective
library from source. Fortunately the list of LLDB dependencies is not big and
if you are only interested in the server component, you can reduce this even
further by passing the appropriate cmake options, such as:

::

  -DLLDB_DISABLE_LIBEDIT=1
  -DLLDB_DISABLE_CURSES=1
  -DLLDB_DISABLE_PYTHON=1
  -DLLVM_ENABLE_TERMINFO=0

In this case you, will often not need anything other than the standard C and
C++ libraries.

Once all of the dependencies are in place, it's just a matter of configuring
the build system with the locations and arguments of all the necessary tools.
The most important cmake options here are:

* ``CMAKE_CROSSCOMPILING`` : Set to 1 to enable cross-compilation.
* ``CMAKE_LIBRARY_ARCHITECTURE`` : Affects the cmake search path when looking
  for libraries. You may need to set this to your architecture triple if you do
  not specify all your include and library paths explicitly.
* ``CMAKE_C_COMPILER``, ``CMAKE_CXX_COMPILER`` : C and C++ compilers for the
  target architecture
* ``CMAKE_C_FLAGS``, ``CMAKE_CXX_FLAGS`` : The flags for the C and C++ target
  compilers. You may need to specify the exact target cpu and abi besides the
  include paths for the target headers.
* ``CMAKE_EXE_LINKER_FLAGS`` : The flags to be passed to the linker. Usually
  just a list of library search paths referencing the target libraries.
* ``LLVM_TABLEGEN``, ``CLANG_TABLEGEN`` : Paths to llvm-tblgen and clang-tblgen
  for the host architecture. If you already have built clang for the host, you
  can point these variables to the executables in your build directory. If not,
  you will need to build the llvm-tblgen and clang-tblgen host targets at
  least.
* ``LLVM_HOST_TRIPLE`` : The triple of the system that lldb (or lldb-server)
  will run on. Not setting this (or setting it incorrectly) can cause a lot of
  issues with remote debugging as a lot of the choices lldb makes depend on the
  triple reported by the remote platform.

You can of course also specify the usual cmake options like
``CMAKE_BUILD_TYPE``, etc.

**Example 1: Cross-compiling for linux arm64 on Ubuntu host**

Ubuntu already provides the packages necessary to cross-compile LLDB for arm64.
It is sufficient to install packages ``gcc-aarch64-linux-gnu``,
``g++-aarch64-linux-gnu``, ``binutils-aarch64-linux-gnu``. Then it is possible
to prepare the cmake build with the following parameters:

::

  -DCMAKE_CROSSCOMPILING=1 \
  -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
  -DLLVM_HOST_TRIPLE=aarch64-unknown-linux-gnu \
  -DLLVM_TABLEGEN=<path-to-host>/bin/llvm-tblgen \
  -DCLANG_TABLEGEN=<path-to-host>/bin/clang-tblgen \
  -DLLDB_DISABLE_PYTHON=1 \
  -DLLDB_DISABLE_LIBEDIT=1 \
  -DLLDB_DISABLE_CURSES=1

An alternative (and recommended) way to compile LLDB is with clang.
Unfortunately, clang is not able to find all the include paths necessary for a
successful cross-compile, so we need to help it with a couple of CFLAGS
options. In my case it was sufficient to add the following arguments to
``CMAKE_C_FLAGS`` and ``CMAKE_CXX_FLAGS`` (in addition to changing
``CMAKE_C(XX)_COMPILER`` to point to clang compilers):

::

  -target aarch64-linux-gnu \
  -I /usr/aarch64-linux-gnu/include/c++/4.8.2/aarch64-linux-gnu \
  -I /usr/aarch64-linux-gnu/include

If you wanted to build a full version of LLDB and avoid passing
``-DLLDB_DISABLE_PYTHON`` and other options, you would need to obtain the
target versions of the respective libraries. The easiest way to achieve this is
to use the qemu-debootstrap utility, which can prepare a system image using
qemu and chroot to simulate the target environment. Then you can install the
necessary packages in this environment (python-dev, libedit-dev, etc.) and
point your compiler to use them using the correct -I and -L arguments.

**Example 2: Cross-compiling for Android on Linux**

In the case of Android, the toolchain and all required headers and libraries
are available in the Android NDK.

The NDK also contains a cmake toolchain file, which makes configuring the build
much simpler. The compiler, include and library paths will be configured by the
toolchain file and all you need to do is to select the architecture
(ANDROID_ABI) and platform level (``ANDROID_PLATFORM``, should be at least 21).
You will also need to set ``ANDROID_ALLOW_UNDEFINED_SYMBOLS=On``, as the
toolchain file defaults to "no undefined symbols in shared libraries", which is
not compatible with some llvm libraries. The first version of NDK which
supports this approach is r14.

For example, the following arguments are sufficient to configure an android
arm64 build:

::

  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-21 \
  -DANDROID_ALLOW_UNDEFINED_SYMBOLS=On \
  -DLLVM_HOST_TRIPLE=aarch64-unknown-linux-android \
  -DCROSS_TOOLCHAIN_FLAGS_NATIVE='-DCMAKE_C_COMPILER=cc;-DCMAKE_CXX_COMPILER=c++'

Note that currently only lldb-server is functional on android. The lldb client
is not supported and unlikely to work.

Building The Documentation
--------------------------

If you wish to build the optional (reference) documentation, additional
dependencies are required:

* Sphinx (for the website)
* Graphviz (for the 'dot' tool)
* doxygen (if you wish to build the C++ API reference)
* epydoc (if you wish to build the Python API reference)

To install the prerequisites for building the documentation (on Debian/Ubuntu)
do:

::

  > sudo apt-get install doxygen graphviz python3-sphinx
  > sudo pip install epydoc

To build the documentation, build the desired target(s).

::

  > cmake --build . --target docs-lldb-html
  > cmake --build . --target lldb-cpp-doc
  > cmake --build . --target lldb-python-doc
