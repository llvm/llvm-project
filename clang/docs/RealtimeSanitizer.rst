=================
RealtimeSanitizer
=================

.. contents::
   :local:

Introduction
============
RealtimeSanitizer (a.k.a. RTSan) is a real-time safety testing tool for C and
C++ projects. RTSan can be used to detect real-time violations,such as calls to
methods that are not safe for use in functions with deterministic runtime
requirements.

The tool can detect the following types of real-time violations:

* System calls
* Allocations
* Exceptions

These checks are put in place when compiling with the
``-fsanitize=realtime`` flag, for functions marked with
``[[clang::nonblocking]]``.

.. code-block:: c

   void process_audio(float* buffer) [[clang::nonblocking]] {
      ...
   }

The runtime slowdown introduced by RealtimeSanitizer is trivial. Code in
real-time contexts without real-time safety violations have no slowdown.

How to build
============

Build LLVM/Clang with `CMake <https://llvm.org/docs/CMake.html>` and enable the
``compiler-rt`` runtime. An example CMake configuration that will allow for the
use/testing of RealtimeSanitizer:

.. code-block:: console

   $ cmake -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang" -DLLVM_ENABLE_RUNTIMES="compiler-rt" <path to source>/llvm

Usage
=====

There are two requirements:

1. The code must be compiled with the ``-fsanitize=realtime`` flag.
2. Functions that are subject to real-time constraints must be marked
   with the ``[[clang::nonblocking]]`` attribute.

Typically, these attributes should be added onto the functions that are entry
points for threads with real-time priority. These threads are subject to a fixed
callback time, such as audio callback threads or rendering loops in video game
code.

.. code-block:: console

   % cat example_realtime_violation.cpp
   int main() [[clang::nonblocking]] {
     int* p = new int;
     return 0;
   }

   # Compile and link
   % clang -fsanitize=realtime -g example_realtime_violation.cpp

If a real-time safety violation is detected in a ``[[clang::nonblocking]]``
context, or any function invoked by that function, the program will exit with a
non-zero exit code.

.. code-block:: console

   % clang -fsanitize=realtime -g example_realtime_violation.cpp
   % ./a.out
   Real-time violation: intercepted call to real-time unsafe function `malloc` in real-time context! Stack trace:
    #0 0x00010065ad9c in __rtsan::PrintStackTrace() rtsan_stack.cpp:45
    #1 0x00010065abcc in __rtsan::Context::ExpectNotRealtime(char const*) rtsan_context.cpp:78
    #2 0x00010065b8d0 in malloc rtsan_interceptors.cpp:289
    #3 0x000195bd7bd0 in operator new(unsigned long)+0x1c (libc++abi.dylib:arm64+0x16bd0)
    #4 0xb338001000dbf68  (<unknown module>)
    #5 0x0001958960dc  (<unknown module>)
    #6 0x45737ffffffffffc  (<unknown module>)
