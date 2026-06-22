.. title:: clang-tidy - bugprone-std-exception-baseclass

bugprone-std-exception-baseclass
================================

Ensure that every value that in a ``throw`` expression is an instance of
``std::exception``.

Deriving all exceptions from ``std::exception`` allows callers to catch
all exceptions with a single catch block and provides access to the
``what()`` method for diagnostics. Throwing arbitrary types creates
hidden contracts, reduces interoperability with the standard library,
and may result in program termination.

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
