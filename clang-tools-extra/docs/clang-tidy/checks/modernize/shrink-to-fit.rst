.. title:: clang-tidy - modernize-shrink-to-fit

modernize-shrink-to-fit
=======================


Replace copy and swap tricks on shrinkable containers with the
``shrink_to_fit()`` method call.

The ``shrink_to_fit()`` method is more readable and more effective than
the copy and swap trick to reduce the capacity of a shrinkable container.
Note that, the ``shrink_to_fit()`` method is only available in C++11 and up.
