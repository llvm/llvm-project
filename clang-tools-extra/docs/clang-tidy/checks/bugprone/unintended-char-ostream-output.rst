.. title:: clang-tidy - bugprone-unintended-char-ostream-output

bugprone-unintended-char-ostream-output
=======================================

Finds unintended character output from ``unsigned char`` and ``signed char`` to an
``ostream``.

Normally, when ``unsigned char (uint8_t)`` or ``signed char (int8_t)`` is used, it
is more likely a number than a character. However, when it is passed directly to
``std::ostream``'s ``operator<<``, the result is the character output instead
of the numeric value. This often contradicts the developer's intent to print
integer values.

.. code-block:: c++

  uint8_t v = 65;
  std::cout << v; // output 'A' instead of '65'

It could be fixed as

.. code-block:: c++

  std::cout << static_cast<uint32_t>(v);

Or cast to char to explicitly indicate the intent

.. code-block:: c++

  std::cout << static_cast<char>(v);

.. option:: CastTypeName

  When `CastTypeName` is specified, the fix-it will use `CastTypeName` as the
  cast target type. Otherwise, fix-it will automatically infer the type.
