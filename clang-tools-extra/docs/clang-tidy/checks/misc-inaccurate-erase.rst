.. title:: clang-tidy - misc-inaccurate-erase

misc-inaccurate-erase
=====================


Checks for inaccurate use of the ``erase()`` method.

Algorithms like ``remove()`` do not actually remove any element from the
container but return an iterator to the first redundant element at the end
of the container. These redundant elements must be removed using the
``erase()`` method. This check warns when not all of the elements will be
removed due to using an inappropriate overload.
