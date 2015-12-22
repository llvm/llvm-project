.. title:: clang-tidy - readability-container-size-empty

readability-container-size-empty
================================


Checks whether a call to the ``size()`` method can be replaced with a call to
``empty()``.

The emptiness of a container should be checked using the ``empty()`` method
instead of the ``size()`` method. It is not guaranteed that ``size()`` is a
constant-time function, and it is generally more efficient and also shows
clearer intent to use ``empty()``. Furthermore some containers may implement
the ``empty()`` method but not implement the ``size()`` method. Using ``empty()``
whenever possible makes it easier to switch to another container in the
future.
