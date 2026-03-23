.. title:: clang-tidy - modernize-use-std-bit

modernize-use-std-bit
=====================

Finds common idioms which can be replaced by standard functions from the
``<bit>`` C++20 header.

Covered scenarios:

============================== =======================
Expression                     Replacement
------------------------------ -----------------------
``x && !(x & (x - 1))``        ``std::has_one_bit(x)``
``(x != 0) && !(x & (x - 1))`` ``std::has_one_bit(x)``
``(x > 0) && !(x & (x - 1))``  ``std::has_one_bit(x)``
``std::bitset<N>(x).count()``  ``std::popcount(x)``
============================== =======================
