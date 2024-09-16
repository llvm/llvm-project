=================
RealtimeSanitizer
=================

.. contents::
   :local:

Introduction
============
RealtimeSanitizer (a.k.a. RTSan) is a real-time safety testing tool for C and C++
projects. RTSan can be used to detect real-time violations, i.e. calls to methods
that are not safe for use in functions with deterministic runtime requirements.
RTSan considers any function marked with the ``[[clang::nonblocking]]`` attribute
to be a real-time function. If RTSan detects a call to ``malloc``, ``free``,
``pthread_mutex_lock``, or anything else that could have a non-deterministic
execution time in a function marked ``[[clang::nonblocking]]``
RTSan raises an error.

The runtime slowdown introduced by RealtimeSanitizer is negligible.

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
   #include <vector>

   void violation() [[clang::nonblocking]]{
     std::vector<float> v;
     v.resize(100);
   }

   int main() {
     violation();
     return 0;
   }
   # Compile and link
   % clang++ -fsanitize=realtime -g example_realtime_violation.cpp

If a real-time safety violation is detected in a ``[[clang::nonblocking]]``
context, or any function invoked by that function, the program will exit with a
non-zero exit code.

.. code-block:: console

   % clang++ -fsanitize=realtime -g example_realtime_violation.cpp
   % ./a.out
   Real-time violation: intercepted call to real-time unsafe function `malloc` in real-time context! Stack trace:
    #0 0x000102893034 in __rtsan::PrintStackTrace() rtsan_stack.cpp:45
    #1 0x000102892e64 in __rtsan::Context::ExpectNotRealtime(char const*) rtsan_context.cpp:78
    #2 0x00010289397c in malloc rtsan_interceptors.cpp:286
    #3 0x000195bd7bd0 in operator new(unsigned long)+0x1c (libc++abi.dylib:arm64+0x16bd0)
    #4 0x5c7f00010230f07c  (<unknown module>)
    #5 0x00010230f058 in std::__1::__libcpp_allocate[abi:ue170006](unsigned long, unsigned long) new:324
    #6 0x00010230effc in std::__1::allocator<float>::allocate[abi:ue170006](unsigned long) allocator.h:114
    ... snip ...
    #10 0x00010230e4bc in std::__1::vector<float, std::__1::allocator<float>>::__append(unsigned long) vector:1162
    #11 0x00010230dcdc in std::__1::vector<float, std::__1::allocator<float>>::resize(unsigned long) vector:1981
    #12 0x00010230dc28 in violation() main.cpp:5
    #13 0x00010230dd64 in main main.cpp:9
    #14 0x0001958960dc  (<unknown module>)
    #15 0x2f557ffffffffffc  (<unknown module>)

Disabling
---------

In some circumstances, you may want to suppress error reporting in a specific scope.

In C++, this is achieved via  ``__rtsan::ScopedDisabler``. Within the scope where the ``ScopedDisabler`` object is instantiated, all sanitizer error reports are suppressed. This suppression applies to the current scope as well as all invoked functions, including any functions called transitively. 

.. code-block:: c++

    #include <sanitizer/rtsan_interface.h>

    void process(const std::vector<float>& buffer) [[clang::nonblocking]] {
        {
            __rtsan::ScopedDisabler d;
            ...
        }
    }

If RealtimeSanitizer is not enabled at compile time (i.e., the code is not compiled with the ``-fsanitize=realtime`` flag), the ``ScopedDisabler`` is compiled as a no-op.

In C, you can use the ``__rtsan_disable()`` and ``rtsan_enable()`` functions to manually disable and re-enable RealtimeSanitizer checks. 

.. code-block:: c++

    #include <sanitizer/rtsan_interface.h>

    int process(const float* buffer) [[clang::nonblocking]]
    {
        {
            __rtsan_disable();

            ...

            __rtsan_enable();
        }
    }

Each call to ``__rtsan_disable()`` must be paired with a subsequent call to ``__rtsan_enable()`` to restore normal sanitizer functionality. If a corresponding ``rtsan_enable()`` call is not made, the behavior is undefined.

Compile-time sanitizer detection
--------------------------------

Clang provides the pre-processor macro ``__has_feature`` which may be used to detect if RealtimeSanitizer is enabled at compile-time.

.. code-block:: c++

    #if defined(__has_feature) && __has_feature(realtime_sanitizer)
    ...
    #endif
