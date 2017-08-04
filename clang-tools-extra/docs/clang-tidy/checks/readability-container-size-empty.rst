.. title:: clang-tidy - readability-container-size-empty

readability-container-size-empty
================================


Checks whether a call to the ``size()`` method can be replaced with a call to
``empty()``.

The emptiness of a container should be checked using the ``empty()`` method
instead of the ``size()`` method. It is not guaranteed that ``size()`` is a
constant-time function, and it is generally more efficient and also shows
clearer intent to use ``empty()``. Furthermore some containers may implement
the ``empty()`` method but not implement the ``size()`` method. Using
``empty()`` whenever possible makes it easier to switch to another container in
the future.

The check issues warning if a container has ``size()`` and ``empty()`` methods
matching following signatures:

.. code-block:: c++

  size_type size() const;
  bool empty() const;

`size_type` can be any kind of integer type.
