.. title:: clang-tidy - bugprone-std-exception-baseclass

bugprone-std-exception-baseclass
================================

Ensure that every value that in a ``throw`` expression is an instance of
``std::exception``.

Exception types should inherit from ``std::exception`` so they can be
handled consistently and caught as ``std::exception``.
Exception objects exist to propagate error information
and must not be created without being thrown.

.. code-block:: c++

  class custom_exception {};

  void throwing() noexcept(false) {
    // Problematic throw expressions.
    throw int(42);
    throw custom_exception();
  }

  class mathematical_error : public std::exception {};

  void throwing2() noexcept(false) {
    // These kind of throws are ok.
    throw mathematical_error();
    throw std::runtime_error();
    throw std::exception();
  }
