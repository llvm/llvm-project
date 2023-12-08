.. title:: clang-tidy - abseil-str-cat-append

abseil-str-cat-append
=====================

Flags uses of ``absl::StrCat()`` to append to a ``std::string``. Suggests
``absl::StrAppend()`` should be used instead.

The extra calls cause unnecessary temporary strings to be constructed. Removing
them makes the code smaller and faster.

.. code-block:: c++

  a = absl::StrCat(a, b); // Use absl::StrAppend(&a, b) instead.

Does not diagnose cases where ``absl::StrCat()`` is used as a template
argument for a functor.
