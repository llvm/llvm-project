.. title:: clang-tidy - bugprone-capture-this-by-field

bugprone-capture-this-by-field
==============================

Finds lambda captures that capture the ``this`` pointer and store it as class
members without handle the copy and move constructors and the assignments.

Capture this in a lambda and store it as a class member is dangerous because the
lambda can outlive the object it captures. Especially when the object is copied
or moved, the captured ``this`` pointer will be implicitly propagated to the
new object. Most of the time, people will believe that the captured ``this``
pointer points to the new object, which will lead to bugs.


.. code-block:: c++

  struct C {
    C() : Captured([this]() -> C const * { return this; }) {}
    std::function<C const *()> Captured;
  };

  void foo() {
    C v1{};
    C v2 = v1; // v2.Captured capture v1's this pointer
    assert(v2.Captured() == v1.Captured()); // v2.Captured capture v1's this pointer
    assert(v2.Captured() == &v2); // assertion failed.
  }
