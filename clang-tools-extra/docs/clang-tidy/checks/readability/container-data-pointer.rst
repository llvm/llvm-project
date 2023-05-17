.. title:: clang-tidy - readability-container-data-pointer

readability-container-data-pointer
==================================

Finds cases where code could use ``data()`` rather than the address of the
element at index 0 in a container. This pattern is commonly used to materialize
a pointer to the backing data of a container. ``std::vector`` and
``std::string`` provide a ``data()`` accessor to retrieve the data pointer which
should be preferred.

This also ensures that in the case that the container is empty, the data pointer
access does not perform an errant memory access.
