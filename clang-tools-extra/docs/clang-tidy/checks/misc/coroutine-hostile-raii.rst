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

Exclusions
-------
It is possible to make the check treat certain suspensions as safe.
``co_await``-ing an expression of ``awaitable`` type is considered
safe if the ``awaitable`` type is annotated with 
``[[clang::annotate("coro_raii_safe_suspend")]]``.
RAII objects persisting across such a ``co_await`` expression are
considered safe and hence are not flagged.

This annotation can be used to mark ``awaitable`` types which can be safely
awaited while having hostile RAII objects in scope. For example, such safe
``awaitable`` could ensure resumption on the same thread or even unlock the mutex
on suspension and reacquire on resumption.

Example usage:

.. code-block:: c++

  struct [[clang::annotate("coro_raii_safe_suspend")]] safe_awaitable {
    bool await_ready() noexcept { return false; }
    void await_suspend(std::coroutine_handle<>) noexcept {}
    void await_resume() noexcept {}
  };

  task coro() {
    const std::lock_guard l(&mu_);
    co_await safe_awaitable{};
  }

  auto wait() { return safe_awaitable{}; }

  task coro() {
    const std::lock_guard l(&mu_); // No warning.
    co_await safe_awaitable{};
    co_await wait();
  }

Options
-------

.. option:: RAIITypesList

    A semicolon-separated list of qualified types which should not be allowed to
    persist across suspension points.
    Eg: ``my::lockable; a::b;::my::other::lockable;``
    The default value of this option is `"std::lock_guard;std::scoped_lock"`.
