.. title:: clang-tidy - cppcoreguidelines-missing-std-forward

cppcoreguidelines-missing-std-forward
=====================================

Warns when a forwarding reference parameter is not forwarded inside the
function body.

Example:

.. code-block:: c++

  template <class T>
  void wrapper(T&& t) {
    impl(std::forward<T>(t), 1, 2); // Correct
  }

  template <class T>
  void wrapper2(T&& t) {
    impl(t, 1, 2); // Oops - should use std::forward<T>(t)
  }

  template <class T>
  void wrapper3(T&& t) {
    impl(std::move(t), 1, 2); // Also buggy - should use std::forward<T>(t)
  }

  template <class F>
  void wrapper_function(F&& f) {
    std::forward<F>(f)(1, 2); // Correct
  }

  template <class F>
  void wrapper_function2(F&& f) {
    f(1, 2); // Incorrect - may not invoke the desired qualified function operator
  }

This check implements `F.19
<http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rf-forward>`_
from the C++ Core Guidelines.
