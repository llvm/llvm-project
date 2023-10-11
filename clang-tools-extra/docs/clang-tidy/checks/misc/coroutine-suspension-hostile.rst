.. title:: clang-tidy - misc-coroutine-hostile-raii

misc-coroutine-hostile-raii
====================

This check detects hostile-RAII objects which should not persist across a suspension point in a coroutine.
Since after a suspension a coroutine could potentially be resumed on a different thread,
such RAII objects could be created by one thread but destroyed by another.
Certain RAII types could be hostile to being destroyed by another thread.

The check considers the following type as hostile:

 - Scoped-lockable types: A scoped-lockable object persisting across a suspension point in a coroutine is 
 problematic as it is possible for the lock held by this object at the suspension 
 point to be unlocked by a different thread. This would be undefined behaviour.

 - Types belonging to a configurable denylist.

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

.. option:: RAIIDenyList

   A semicolon-separated list of qualified types which should not be allowed to persist across suspension points.
   Do not include trailing `::` in the qualified name.
   Eg: `my::lockable; my::other::lockable;`