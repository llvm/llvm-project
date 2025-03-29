.. title:: clang-tidy - misc-coroutine-hostile-raii

misc-coroutine-hostile-raii
===========================

Detects when objects of certain hostile RAII types persists across suspension
points in a coroutine. Such hostile types include scoped-lockable types and
types belonging to a configurable denylist.

Some objects require that they be destroyed on the same thread that created them.
Traditionally this requirement was often phrased as "must be a local variable",
under the assumption that local variables always work this way. However this is
incorrect with C++20 coroutines, since an intervening ``co_await`` may cause the
coroutine to suspend and later be resumed on another thread.

The lifetime of an object that requires being destroyed on the same thread must
not encompass a ``co_await`` or ``co_yield`` point. If you create/destroy an object,
you must do so without allowing the coroutine to suspend in the meantime.

Following types are considered as hostile:

 - Scoped-lockable types: A scoped-lockable object persisting across a suspension
   point is problematic as the lock held by this object could be unlocked by a
   different thread. This would be undefined behaviour.
   This includes all types annotated with the ``scoped_lockable`` attribute.

 - Types belonging to a configurable denylist.

.. code-block:: c++

  // Call some async API while holding a lock.
  task coro() {
    const std::lock_guard l(&mu_);

    // Oops! The async Bar function may finish on a different
    // thread from the one that created the lock_guard (and called
    // Mutex::Lock). After suspension, Mutex::Unlock will be called on the wrong thread.
    co_await Bar();
  }

Options
-------

.. option:: RAIITypesList

    A semicolon-separated list of qualified types which should not be allowed to
    persist across suspension points.
    Eg: ``my::lockable; a::b;::my::other::lockable;``
    The default value of this option is `"std::lock_guard;std::scoped_lock"`.

.. option:: AllowedAwaitablesList

    A semicolon-separated list of qualified types of awaitables types which can
    be safely awaited while having hostile RAII objects in scope.

    ``co_await``-ing an expression of ``awaitable`` type is considered
    safe if the ``awaitable`` type is part of this list.
    RAII objects persisting across such a ``co_await`` expression are
    considered safe and hence are not flagged.

    Example usage:

    .. code-block:: c++

      // Consider option AllowedAwaitablesList = "safe_awaitable"
      struct safe_awaitable {
        bool await_ready() noexcept { return false; }
        void await_suspend(std::coroutine_handle<>) noexcept {}
        void await_resume() noexcept {}
      };
      auto wait() { return safe_awaitable{}; }

      task coro() {
        // This persists across both the co_await's but is not flagged
        // because the awaitable is considered safe to await on.
        const std::lock_guard l(&mu_);
        co_await safe_awaitable{};
        co_await wait();
      }

    Eg: ``my::safe::awaitable;other::awaitable``
    The default value of this option is empty string `""`.

