.. title:: clang-tidy - modernize-min-max-use-initializer-list

modernize-min-max-use-initializer-list
======================================

Replaces nested ``std::min`` and ``std::max`` calls with an initializer list 
where applicable.

For instance, consider the following code:

.. code-block:: cpp

  int a = std::max(std::max(i, j), k);

The check will transform the above code to:

.. code-block:: cpp

  int a = std::max({i, j, k});

Performance Considerations
==========================

While this check simplifies the code and makes it more readable, it may cause 
performance degradation for non-trivial types due to the need to copy objects 
into the initializer list.

To avoid this, it is recommended to use `std::ref` or `std::cref` for
non-trivial types:

.. code-block:: cpp

  std::string b = std::max({std::ref(i), std::ref(j), std::ref(k)});

Options
=======

.. option:: IncludeStyle

  A string specifying which include-style is used, `llvm` or `google`. Default
  is `llvm`.

.. option:: IgnoreNonTrivialTypes

  A boolean specifying whether to ignore non-trivial types. Default is `true`.

.. option:: IgnoreTrivialTypesOfSizeAbove

  An integer specifying the size (in bytes) above which trivial types are
  ignored. Default is `32`.