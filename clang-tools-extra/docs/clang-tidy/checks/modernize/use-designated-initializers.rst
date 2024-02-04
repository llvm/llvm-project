.. title:: clang-tidy - modernize-use-designated-initializers

modernize-use-designated-initializers
=====================================

Finds initializer lists for aggregate type that could be written as
designated initializers instead.

With plain initializer lists, it is very easy to introduce bugs when adding
new fields in the middle of a struct or class type. The same confusion might
arise when changing the order of fields.

C++ 20 supports the designated initializer syntax for aggregate types.
By applying it, we can always be sure that aggregates are constructed correctly,
because every variable being initialized is referenced by name.

Even when compiling in a language version older than C++ 20, depending on you compiler,
designated initializers are potentially supported. Therefore, the check is not restricted
to C++ 20 and older.

Example:

.. code-block::

    struct S { int i, j; };

is an aggregate type that should be initialized as

.. code-block::

    S s{.i = 1, .j = 2};

instead of

.. code-block::

    S s{1, 2};

which could easily become an issue when ``i`` and ``j`` are swapped in the
declaration of ``S``.

Options
-------

.. option:: IgnoreSingleElementAggregates

    The value `false` specifies that even initializers for aggregate types
    with only a single element should be checked. The default value is `true`.

.. option:: RestrictToPODTypes

    The value `true` specifies that only Plain Old Data (POD) types shall be
    checked. This makes the check applicable to even older C++ standards.
    The default value is `false`.
