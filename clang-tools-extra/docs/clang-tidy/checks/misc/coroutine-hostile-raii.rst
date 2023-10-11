.. title:: clang-tidy - misc-coroutine-hostile-raii

misc-coroutine-hostile-raii
====================

This check detects hostile-RAII objects which should not persist across a 
suspension point in a coroutine.

Some objects require that they be destroyed on the same thread that created them. 
Traditionally this requirement was often phrased as "must be a local variable",
under the assumption that local variables always work this way. However this is
incorrect with C++20 coroutines, since an intervening `co_await` may cause the
coroutine to suspend and later be resumed on another thread.

The lifetime of an object that requires being destroyed on the same thread must 
not encompass a `co_await` or `co_yield` point. If you create/destroy an object,
you must do so without allowing the coroutine to suspend in the meantime.

The check considers the following type as hostile:

 - Scoped-lockable types: A scoped-lockable object persisting across a suspension
 point is problematic as the lock held by this object could be unlocked by a 
 different thread. This would be undefined behaviour.

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

   A semicolon-separated list of qualified types which should not be allowed to 
   persist across suspension points.
   Do not include `::` in the beginning of the qualified name.
   Eg: `my::lockable; a::b; ::my::other::lockable;`