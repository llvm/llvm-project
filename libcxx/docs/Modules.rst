.. _ModulesInLibcxx:

=================
Modules in libc++
=================

.. warning:: Modules are an experimental feature. It has additional build
             requirements and not all libc++ configurations are supported yet.

             The work is still in an early developement state and not
             considered stable nor complete

This page contains information regarding C++23 module support in libc++.
There are two kinds of modules available in Clang

 * `Clang specific modules <https://clang.llvm.org/docs/Modules.html>`_
 * `C++ modules <https://clang.llvm.org/docs/StandardCPlusPlusModules.html>`_

This page mainly discusses the C++ modules. In C++20 there are also header units,
these are not part of this document.

Overview
========

The module sources are stored in ``.cppm`` files. Modules need to be available
as BMIs, which are ``.pcm`` files for Clang. BMIs are not portable, they depend
on the compiler used and its compilation flags. Therefore there needs to be a
way to distribute the ``.cppm`` files to the user and offer a way for them to
build and use the ``.pcm`` files. It is expected this will be done by build
systems in the future. To aid early adaptor and build system vendors libc++
currently ships a CMake project to aid building modules.

.. note:: This CMake file is intended to be a temporary solution and will
          be removed in the future. The timeline for the removal depends
          on the availability of build systems with proper module support.

What works
~~~~~~~~~~

 * Building BMIs
 * Running tests using the ``std`` module
 * Using the ``std`` module in external projects
 * The following "parts disabled" configuration options are supported

   * ``LIBCXX_ENABLE_LOCALIZATION``
   * ``LIBCXX_ENABLE_WIDE_CHARACTERS``

Some of the current limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 * There is no official build system support, libc++ has experimental CMake support
 * Requires CMake 3.26
 * Requires Ninja 1.11
 * Requires a recent Clang 17
 * The path to the compiler may not be a symlink, ``clang-scan-deps`` does
   not handle that case properly
 * Only C++23 is tested
 * Libc++ is not tested with modules instead of headers
 * The module ``.cppm`` files are not installed
 * The experimental ``PSTL`` library is not supported
 * Clang supports modules using GNU extensions, but libc++ does not work using
   GNU extensions.
 * Clang:
    * Including headers after importing the ``std`` module may fail. This is
      hard to solve and there is a work-around by first including all headers
      `bug report <https://github.com/llvm/llvm-project/issues/61465>`__.

Blockers
~~~~~~~~

  * libc++

    * Currently the tests only test with modules enabled, but do not import
      modules instead of headers. When converting tests to using modules there
      are still failures. These are under investigation.

    * It has not been determined how to fully test libc++ with modules instead
      of headers.

  * Clang

    * Some concepts do not work properly
      `bug report <https://github.com/llvm/llvm-project/issues/62943>`__.


Using in external projects
==========================

Users need to be able to build their own BMI files.

.. note:: The requirements for users to build their own BMI files will remain
   true for the forseeable future. For now this needs to be done manually.
   Once libc++'s implementation is more mature we will reach out to build
   system vendors, with the goal that building the BMI files is done by
   the build system.

Currently this requires a local build of libc++ with modules enabled. Since
modules are not part of the installation yet, they are used from the build
directory. First libc++ needs to be build with module support enabled.

.. code-block:: bash

  $ git clone https://github.com/llvm/llvm-project.git
  $ cd llvm-project
  $ mkdir build
  $ cmake -G Ninja -S runtimes -B build -DLIBCXX_ENABLE_STD_MODULES=ON -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;libunwind"
  $ ninja -C build

The above ``build`` directory will be referred to as ``<build>`` in the
rest of these instructions.

This is a small sample program that uses the module ``std``. It consists of a
``CMakeLists.txt`` and a ``main.cpp`` file.

.. code-block:: cpp

  import std;

  int main() { std::cout << "Hello modular world\n"; }

.. code-block:: cmake

  cmake_minimum_required(VERSION 3.26.0 FATAL_ERROR)
  project("module"
    LANGUAGES CXX
  )

  #
  # Set language version used
  #

  # At the moment only C++23 is tested.
  set(CMAKE_CXX_STANDARD 23)
  set(CMAKE_CXX_STANDARD_REQUIRED YES)
  # Libc++ doesn't support compiler extensions for modules.
  set(CMAKE_CXX_EXTENSIONS OFF)

  #
  # Enable modules in CMake
  #

  # This is required to write your own modules in your project.
  if(CMAKE_VERSION VERSION_LESS "3.27.0")
    set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API "2182bf5c-ef0d-489a-91da-49dbc3090d2a")
  else()
    set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API "aa1f7df0-828a-4fcd-9afc-2dc80491aca7")
  endif()
  set(CMAKE_EXPERIMENTAL_CXX_MODULE_DYNDEP 1)

  #
  # Import the modules from libc++
  #

  include(FetchContent)
  FetchContent_Declare(
    std
    URL "file://${LIBCXX_BUILD}/modules/c++/v1/"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_GetProperties(std)
  if(NOT std_POPULATED)
    FetchContent_Populate(std)
    add_subdirectory(${std_SOURCE_DIR} ${std_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()

  #
  # Adjust project compiler flags
  #

  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fprebuilt-module-path=${CMAKE_BINARY_DIR}/_deps/std-build/CMakeFiles/std.dir/>)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-nostdinc++>)
  # The include path needs to be set to be able to use macros from headers.
  # For example from, the headers <cassert> and <version>.
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-isystem>)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:${LIBCXX_BUILD}/include/c++/v1>)

  #
  # Adjust project linker flags
  #

  add_link_options($<$<COMPILE_LANGUAGE:CXX>:-nostdlib++>)
  add_link_options($<$<COMPILE_LANGUAGE:CXX>:-L${LIBCXX_BUILD}/lib>)
  add_link_options($<$<COMPILE_LANGUAGE:CXX>:-Wl,-rpath,${LIBCXX_BUILD}/lib>)
  # Linking against std is required for CMake to get the proper dependencies
  link_libraries(std c++)

  #
  # Add the project
  #

  add_executable(main)
  target_sources(main
    PRIVATE
      main.cpp
  )

Building this project is done with the following steps, assuming the files
``main.cpp`` and ``CMakeLists.txt`` are copied in the current directory.

.. code-block:: bash

  $ mkdir build
  $ cmake -G Ninja -S . -B build -DCMAKE_CXX_COMPILER=<path-to-compiler> -DLIBCXX_BUILD=<build>
  $ ninja -C build
  $ build/main

.. warning:: ``<path-to-compiler>`` should point point to the real binary and
             not to a symlink.

.. warning:: When using these examples in your own projects make sure the
             compilation flags are the same for the ``std`` module and your
             project. Some flags will affect the generated code, when these
             are different the module cannot be used. For example using
             ``-pthread`` in your project and not in the module will give
             errors like

             ``error: POSIX thread support was disabled in PCH file but is currently enabled``

             ``error: module file _deps/std-build/CMakeFiles/std.dir/std.pcm cannot be loaded due to a configuration mismatch with the current compilation [-Wmodule-file-config-mismatch]``

If you have questions about modules feel free to ask them in the ``#libcxx``
channel on `LLVM's Discord server <https://discord.gg/jzUbyP26tQ>`__.

If you think you've found a bug please it using the `LLVM bug tracker
<https://github.com/llvm/llvm-project/issues>`_. Please make sure the issue
you found is not one of the known bugs or limitations on this page.
