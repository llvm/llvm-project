.. title:: clang-tidy - cppcoreguidelines-avoid-reference-coroutine-parameters

cppcoreguidelines-avoid-reference-coroutine-parameters
======================================================

Warns when a coroutine accepts reference parameters. After a coroutine suspend point,
references could be dangling and no longer valid. Instead, pass parameters as values.

Examples:

.. code-block:: c++

  std::future<int> someCoroutine(int& val) {
    co_await ...;
    // When the coroutine is resumed, 'val' might no longer be valid.
    if (val) ...
  }

This check implements `CP.53
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rcoro-reference-parameters>`_
from the C++ Core Guidelines.
