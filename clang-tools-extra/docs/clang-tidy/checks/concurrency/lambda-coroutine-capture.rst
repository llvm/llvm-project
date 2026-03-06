.. title:: clang-tidy - concurrency-lambda-coroutine-capture

concurrency-lambda-coroutine-capture
====================================

Finds lambda coroutines that capture variables without using the C++23
"deducing this" (explicit object parameter) syntax, which can lead to
use-after-free bugs.

The C++ Core Guidelines have described the problem with lambda coroutines
with non-empty capture lists in `CP.51
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#rcoro-capture>`_.
However, the C++23 explicit object parameter (``this auto``) solves the
issue of the closure object being destructed before the lambda coroutine
resumes by moving captures directly into the coroutine frame, decoupling their
lifetime from the lambda object.

Example
-------

Before:

.. code-block:: c++

  auto handler = [data](int x) -> task {
    co_await process(data, x);  // 'data' may be dangling
  };

After:

.. code-block:: c++

  auto handler = [data](this auto, int x) -> task {
    co_await process(data, x);  // captures live in the coroutine frame
  };

Options
-------

This check has no configurable options.

This check requires C++23 or later (``-std=c++23``). For pre-C++23 code, see
:doc:`cppcoreguidelines-avoid-capturing-lambda-coroutines
<../cppcoreguidelines/avoid-capturing-lambda-coroutines>` which flags the same
problem but recommends avoiding captures entirely rather than using deducing
this.
