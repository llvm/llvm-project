.. title:: clang-tidy - readability-qualified-auto

readability-qualified-auto
==========================

Adds pointer and ``const`` qualifications to ``auto``-typed variables that are deduced
to pointers and ``const`` pointers.

`LLVM Coding Standards <https://llvm.org/docs/CodingStandards.html>`_ advises to
make it obvious if a ``auto`` typed variable is a pointer, constant pointer or 
constant reference. This check will transform ``auto`` to ``auto *`` when the 
type is deduced to be a pointer, as well as adding ``const`` when applicable to
``auto`` pointers or references

.. code-block:: c++

  for (auto &Data : MutatableContainer) {
    change(Data);
  }
  for (auto &Data : ConstantContainer) {
    observe(Data);
  }
  for (auto Data : MutatablePtrContainer) {
    change(*Data);
  }
  for (auto Data : ConstantPtrContainer) {
    observe(*Data);
  }

Would be transformed into:

.. code-block:: c++

  for (auto &Data : MutatableContainer) {
    change(Data);
  }
  for (const auto &Data : ConstantContainer) {
    observe(Data);
  }
  for (auto *Data : MutatablePtrContainer) {
    change(*Data);
  }
  for (const auto *Data : ConstantPtrContainer) {
    observe(*Data);
  }

Note const volatile qualified types will retain their const and volatile qualifiers.

.. code-block:: c++

  const auto Foo = cast<int *>(Baz1);
  const auto Bar = cast<const int *>(Baz2);
  volatile auto FooBar = cast<int*>(Baz3);

Would be transformed into:

.. code-block:: c++

  auto *const Foo = cast<int *>(Baz1);
  const auto *const Bar = cast<const int *>(Baz2);
  auto *volatile FooBar = cast<int*>(Baz3);

This check helps to enforce this `LLVM Coding Standards recommendation
<https://llvm.org/docs/CodingStandards.html#beware-unnecessary-copies-with-auto>`_.
