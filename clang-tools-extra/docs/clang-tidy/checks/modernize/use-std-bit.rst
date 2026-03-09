.. title:: clang-tidy - modernize-use-std-bit

modernize-use-std-bit
=====================

Find common idioms which can be replaced by standrad functions from the
``<bit>`` C++20 header.

.. code-block:: c++

    bool has_one_bit = x && !(x & (x - 1));

    // transforms to

    #include <bit>

    bool has_one_bit = std::has_one_bit(x);
