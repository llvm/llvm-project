.. title:: clang-tidy - bugprone-bit-cast-pointers

bugprone-bit-cast-pointers
==========================

Warns about usage of ``std::bit_cast`` when either the input or output types
are pointers.

The motivation is that ``std::bit_cast`` is advertised as the safe alternative
to type punning via ``reinterpret_cast`` in modern C++. However, one should not
blindly replace ``reinterpret_cast`` with ``std::bit_cast``, as follows:

.. code-block:: c++

    int x{};
    -float y = *reinterpret_cast<float*>(&x);
    +float y = *std::bit_cast<float*>(&x);

The drop-in replacement behaves exactly the same as ``reinterpret_cast``, and
Undefined Behavior is still invoked. ``std::bit_cast`` is copying the bytes of
the input pointer, not the pointee, into an output pointer of a different type,
which violates the strict aliasing rules. However, simply looking at the code,
it looks "safe", because it uses ``std::bit_cast`` which is advertised as safe.

The solution to safe type punning is to apply ``std::bit_cast`` on value types,
not on pointer types:

.. code-block:: c++

    int x{};
    float y = std::bit_cast<float>(x);

This way, the bytes of the input object are copied into the output object, which
is safe from Undefined Behavior. Compilers are able to optimize this copy and
generate identical assembly to the original ``reinterpret_cast`` version.

Alternatively, if a cast between pointers is truly wanted, ``reinterpret_cast``
should be used, to clearly convey intent and enable warnings from compilers and
linters, which should be addressed accordingly.
