.. title:: clang-tidy - hicpp-ignored-remove-result

hicpp-ignored-remove-result
===========================

Ensure that the result of ``std::remove``, ``std::remove_if`` and ``std::unique``
are not ignored according to
`rule 17.5.1 <https://www.perforce.com/resources/qac/high-integrity-cpp-coding-standard/standard-library>`_.

The mutating algorithms ``std::remove``, ``std::remove_if`` and both overloads
of ``std::unique`` operate by swapping or moving elements of the range they are
operating over. On completion, they return an iterator to the last valid
element. In the majority of cases the correct behavior is to use this result as
the first operand in a call to ``std::erase``.

This check is a subset of :doc:`bugprone-unused-return-value <../bugprone/unused-return-value>`
and depending on used options it can be superfluous to enable both checks.

Options
-------

.. option:: AllowCastToVoid

   Controls whether casting return values to ``void`` is permitted. Default: `true`.
