======================================
``[[nodiscard]]`` extensions in libc++
======================================

Libc++ adds ``[[nodiscard]]`` to functions in a lot more places than the
standard does. Any applications of ``[[nodiscard]]`` that aren't required by the
standard written as ``_LIBCPP_NODISCARD_EXT`` to make it possible to disable
them. This can be done by defining ``_LIBCPP_DISABLE_NODISCARD_EXT``.

When should ``[[nodiscard]]`` be added to functions?
====================================================

``[[nodiscard]]`` should be applied to functions

- where it is most likely a correctness issue when discarding the return value.
  For example a locking constructor in ``unique_lock``.
- where most likely something similar was meant if the return value is
  discarded. For example ``vector::empty()``, which probably should have
  been ``clear()``.

  This can help spotting bugs easily which otherwise may take a very long time
  to find.

- which return a constant. For example ``numeric_limits::min()``.
- which only observe a value. For example ``string::size()``.

  Code that discards values from these kinds of functions is dead code. It can
  either be removed, or the programmer meant to do something different.

- where discarding the value is most likely a misuse of the function. For
  example ``find``.

  This protects programmers from assuming too much about how the internals of
  a function work, resulting in less code breaking as optimizations are applied.
