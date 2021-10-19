.. title:: clang-tidy - readability-qualified-auto

readability-qualified-auto
==========================

Adds pointer qualifications to ``auto``-typed variables that are deduced to 
pointers.

`LLVM Coding Standards <https://llvm.org/docs/CodingStandards.html#beware-unnecessary-copies-with-auto>`_
advises to make it obvious if a ``auto`` typed variable is a pointer. This 
check will transform ``auto`` to ``auto *`` when the type is deduced to be a
pointer.

.. code-block:: c++

  for (auto Data : MutatablePtrContainer) {
    change(*Data);
  }
  for (auto Data : ConstantPtrContainer) {
    observe(*Data);
  }

Would be transformed into:

.. code-block:: c++

  for (auto *Data : MutatablePtrContainer) {
    change(*Data);
  }
  for (const auto *Data : ConstantPtrContainer) {
    observe(*Data);
  }

Note ``const`` ``volatile`` qualified types will retain their ``const`` and 
``volatile`` qualifiers. Pointers to pointers will not be fully qualified.

.. code-block:: c++

  const auto Foo = cast<int *>(Baz1);
  const auto Bar = cast<const int *>(Baz2);
  volatile auto FooBar = cast<int *>(Baz3);
  auto BarFoo = cast<int **>(Baz4);

Would be transformed into:

.. code-block:: c++

  auto *const Foo = cast<int *>(Baz1);
  const auto *const Bar = cast<const int *>(Baz2);
  auto *volatile FooBar = cast<int *>(Baz3);
  auto *BarFoo = cast<int **>(Baz4);

Options
-------

.. option:: AddConstToQualified
   
   When set to `true` the check will add const qualifiers variables defined as
   ``auto *`` or ``auto &`` when applicable.
   Default value is `true`.

.. code-block:: c++

   auto Foo1 = cast<const int *>(Bar1);
   auto *Foo2 = cast<const int *>(Bar2);
   auto &Foo3 = cast<const int &>(Bar3);

If AddConstToQualified is set to `false`,  it will be transformed into:

.. code-block:: c++

   const auto *Foo1 = cast<const int *>(Bar1);
   auto *Foo2 = cast<const int *>(Bar2);
   auto &Foo3 = cast<const int &>(Bar3);

Otherwise it will be transformed into:

.. code-block:: c++

   const auto *Foo1 = cast<const int *>(Bar1);
   const auto *Foo2 = cast<const int *>(Bar2);
   const auto &Foo3 = cast<const int &>(Bar3);

Note in the LLVM alias, the default value is `false`.
