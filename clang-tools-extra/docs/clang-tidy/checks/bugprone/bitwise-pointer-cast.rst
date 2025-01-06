.. title:: clang-tidy - bugprone-bitwise-pointer-cast

bugprone-bitwise-pointer-cast
=============================

Warns about code that tries to cast between pointers by means of
``std::bit_cast`` or ``memcpy``.

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
which may violate the strict aliasing rules. However, simply looking at the
code, it looks "safe", because it uses ``std::bit_cast`` which is advertised as
safe.

The solution to safe type punning is to apply ``std::bit_cast`` on value types,
not on pointer types:

.. code-block:: c++

    int x{};
    float y = std::bit_cast<float>(x);

This way, the bytes of the input object are copied into the output object, which
is much safer. Do note that Undefined Behavior can still occur, if there is no
value of type ``To`` corresponding to the value representation produced.
Compilers may be able to optimize this copy and generate identical assembly to
the original ``reinterpret_cast`` version.

Code before C++20 may backport ``std::bit_cast`` by means of ``memcpy``, or
simply call ``memcpy`` directly, which is equally problematic. This is also
detected by this check:

.. code-block:: c++

    int* x{};
    float* y{};
    std::memcpy(&y, &x, sizeof(x));

Alternatively, if a cast between pointers is truly wanted, ``reinterpret_cast``
should be used, to clearly convey the intent and enable warnings from compilers
and linters, which should be addressed accordingly.
