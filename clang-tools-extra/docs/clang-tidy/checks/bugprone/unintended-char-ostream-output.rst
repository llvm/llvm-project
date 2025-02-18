.. title:: clang-tidy - bugprone-unintended-char-ostream-output

bugprone-unintended-char-ostream-output
=======================================

Finds unintended character output from `unsigned char` and `signed char` to an
``ostream``.

Normally, when ``unsigned char (uint8_t)`` or ``signed char (int8_t)`` is used, it
is more likely a number than a character. However, when it is passed directly to
``std::ostream``'s ``operator<<``, resulting in character-based output instead
of numeric value. This often contradicts the developer's intent to print
integer values.

.. code-block:: c++

    uint8_t v = 9;
    std::cout << v; // output '\t' instead of '9'

It could be fixed as

.. code-block:: c++

    std::cout << (uint32_t)v;

Or cast to char to explicitly indicate the intent

.. code-block:: c++

    std::cout << (char)v;
