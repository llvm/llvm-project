.. title:: clang-tidy - fuchsia-virtual-inheritance

fuchsia-virtual-inheritance
===========================

Warns if classes are defined with virtual inheritance.

For example, classes should not be defined with virtual inheritance:

.. code-block:: c++

  class B : public virtual A {};   // warning

See the features disallowed in Fuchsia at https://fuchsia.dev/fuchsia-src/development/languages/c-cpp/cxx?hl=en
