.. title:: clang-tidy - misc-coroutine-suspension-hostile

misc-coroutine-suspension-hostile
====================

This check detects when objects of certain hostile types persists across suspension points in a coroutine.
Such hostile types include scoped-lockable types and types belonging to a configurable denylist.

A scoped-lockable object persisting across a suspension point in a coroutine is 
problematic as it is possible for the lock held by this object at the suspension 
point to be unlocked by a wrong thread if the coroutine resume on a different thread.
This would be undefined behaviour.

The check also diagnosis objects persisting across suspension points which belong to a configurable denylist.

.. code-block:: c++

  // Call some async API while holding a lock.
  {
    const my::MutexLock l(&mu_);

    // Oops! The async Bar function may finish on a different
    // thread from the one that created the MutexLock object and therefore called
    // Mutex::Lock -- now Mutex::Unlock will be called on the wrong thread.
    co_await Bar();
  }


Options
-------

.. option:: DenyTypeList

   A semicolon-separated list of qualified types which should not be allowed to persist across suspension points.
   Do not include trailing `::` in the qualified name.
   Eg: `my::lockable; my::other::lockable;`