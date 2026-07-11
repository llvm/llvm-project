.. title:: clang-tidy - cppcoreguidelines-avoid-reference-coroutine-parameters

cppcoreguidelines-avoid-reference-coroutine-parameters
======================================================

Warns when a coroutine accepts reference parameters. After a coroutine suspend
point, references could be dangling and no longer valid. Instead, pass
parameters as values.

Examples:

.. code-block:: c++

  std::future<int> someCoroutine(int& val) {
    co_await ...;
    // When the coroutine is resumed, 'val' might no longer be valid.
    if (val) ...
  }

This check implements `CP.53
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#rcoro-reference-parameters>`_
from the C++ Core Guidelines.

Options
-------

.. option:: AllowedReturnTypes

  A semicolon-separated list of regular expressions matched against the
  coroutine's return type. Coroutines whose return type matches one of the
  expressions are not flagged, even if they accept reference parameters. This
  is useful for task types that make reference parameters safe by
  construction, such as non-copyable, non-movable coroutine types that can
  only be awaited within the full expression that created them, so every
  argument temporary outlives the coroutine.

  Matching is performed against the canonical, fully-qualified return type, so
  type aliases are resolved before matching. For example, given
  ``using MyTask = ns::Co<int>;``, a coroutine returning ``MyTask`` is matched
  by ``ns::Co<int>`` (or a regular expression such as ``ns::Co<.*>``), but not
  by ``MyTask``.

  The default value is an empty string, which preserves the behavior of
  flagging every coroutine with reference parameters.
