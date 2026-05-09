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
``x << 3 | x >> 61``           ``std::rotl(x, 3)``
``x << 61 | x >> 3``           ``std::rotr(x, 3)``
============================== =======================

Options
-------

.. option:: HonorIntPromotion

  When set to ``true`` (default is ``false``), insert explicit cast to make sure the
  type of the substituted expression is unchanged. Example:

  .. code:: c++

    // Return type is deduced as 'int' (not 'unsigned char') due to implicit conversions.
    auto foo(unsigned char x) {
      return x << 3 | x >> 5;
    }

  Becomes:

  .. code:: c++

    #include <bit>

    auto foo(unsigned char x) {
      return static_cast<int>(std::rotl(x, 3));
    }
