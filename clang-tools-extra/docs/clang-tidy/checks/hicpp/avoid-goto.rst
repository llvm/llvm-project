.. title:: clang-tidy - hicpp-avoid-goto
.. meta::
   :http-equiv=refresh: 5;URL=../cppcoreguidelines/avoid-goto.html

hicpp-avoid-goto
================

The `hicpp-avoid-goto` check is an alias to
:doc:`cppcoreguidelines-avoid-goto <../cppcoreguidelines/avoid-goto>`.
Rule `6.3.1 High Integrity C++ <https://www.perforce.com/resources/qac/high-integrity-cpp-coding-standard/statements>`_
requires that ``goto`` only skips parts of a block and is not used for other
reasons.

Both coding guidelines implement the same exception to the usage of ``goto``.
