.. title:: clang-tidy - cppcoreguidelines-no-suspend-with-lock

cppcoreguidelines-no-suspend-with-lock
======================================

Flags coroutines that suspend while a lock guard is in scope at the
suspension point.

When a coroutine suspends, any mutexes held by the coroutine will remain
locked until the coroutine resumes and eventually destructs the lock guard.
This can lead to long periods with a mutex held and runs the risk of deadlock.

Instead, locks should be released before suspending a coroutine.

This check only checks suspending coroutines while a lock_guard is in scope;
it does not consider manual locking or unlocking of mutexes, e.g., through
calls to ``std::mutex::lock()``.

Examples:

.. code-block:: c++

  future bad_coro() {
    std::lock_guard lock{mtx};
    ++some_counter;
    co_await something(); // Suspending while holding a mutex
  }

  future good_coro() {
    {
      std::lock_guard lock{mtx};
      ++some_counter;
    }
    // Destroy the lock_guard to release the mutex before suspending the coroutine
    co_await something(); // Suspending while holding a mutex
  }

This check implements `CP.52
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rcoro-locks>`_
from the C++ Core Guidelines.
