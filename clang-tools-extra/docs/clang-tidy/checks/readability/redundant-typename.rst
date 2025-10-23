.. title:: clang-tidy - readability-redundant-typename

readability-redundant-typename
==============================

Finds redundant uses of the ``typename`` keyword.

``typename`` is redundant in two cases. First, before non-dependent names:

.. code-block:: c++

  /*typename*/ std::vector<int>::size_type size;

And second, since C++20, before dependent names that appear in a context
where only a type is allowed (the following example shows just a few of them):

.. code-block:: c++

  template <typename T>
  using trait = /*typename*/ T::type;

  template <typename T>
  /*typename*/ T::underlying_type as_underlying(T n) {
    return static_cast</*typename*/ T::underlying_type>(n);
  }

  template <typename T>
  struct S {
    /*typename*/ T::type variable;
    /*typename*/ T::type function(/*typename*/ T::type);
  };
