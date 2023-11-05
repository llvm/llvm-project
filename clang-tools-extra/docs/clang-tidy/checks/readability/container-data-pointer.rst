.. title:: clang-tidy - readability-container-data-pointer

readability-container-data-pointer
==================================

Finds cases where code references the address of the element at index 0 in a container and replaces them with calls to ``data()`` or ``c_str()``.

Using ``data()`` or ``c_str()`` is more readable and ensures that if the container is empty, the data pointer
access does not perform an errant memory access.

Options
-------

.. option:: IgnoredContainers

   Semicolon-separated list of containers regexp for which this check won't be
   enforced. Default is `empty`.
