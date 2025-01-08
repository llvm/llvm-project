.. title:: clang-tidy - readability-container-size-empty

readability-container-size-empty
================================


Checks whether a call to the ``size()``/``length()`` method can be replaced
with a call to ``empty()``.

The emptiness of a container should be checked using the ``empty()`` method
instead of the ``size()``/``length()`` method. It shows clearer intent to use
``empty()``. Furthermore some containers may implement the ``empty()`` method
but not implement the ``size()`` or ``length()`` method. Using ``empty()``
whenever possible makes it easier to switch to another container in the future.

The check issues warning if a container has ``empty()`` and ``size()`` or
``length()`` methods matching following signatures:

.. code-block:: c++

  size_type size() const;
  size_type length() const;
  bool empty() const;

`size_type` can be any kind of integer type.

.. option:: ExcludedComparisonTypes

    A semicolon-separated list of class names for which the check will ignore
    comparisons of objects with default-constructed objects of the same type.
    If a class is listed here, the check will not suggest using ``empty()``
    instead of such comparisons for objects of that class.
    Default value is: `::std::array`.
