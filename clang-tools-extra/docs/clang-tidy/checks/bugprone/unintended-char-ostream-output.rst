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

The check will suggest casting the value to an appropriate type to indicate the
intent, by default, it will cast to ``unsigned int`` for ``unsigned char`` and
``int`` for ``signed char``.

.. code-block:: c++

  std::cout << static_cast<unsigned int>(v); // when v is unsigned char
  std::cout << static_cast<int>(v); // when v is signed char

To avoid lengthy cast statements, add prefix ``+`` to the variable can also
suppress warnings because unary expression will promote the value to an ``int``.

.. code-block:: c++

  std::cout << +v;

Or cast to char to explicitly indicate that output should be a character.

.. code-block:: c++

  std::cout << static_cast<char>(v);

Options
-------

.. option:: AllowedTypes

  A semicolon-separated list of type names that will be treated like the ``char``
  type: the check will not report variables declared with with these types or
  explicit cast expressions to these types. Note that this distinguishes type
  aliases from the original type, so specifying e.g. ``unsigned char`` here
  will not suppress reports about ``uint8_t`` even if it is defined as a
  ``typedef`` alias for ``unsigned char``.
  Default is `unsigned char;signed char`.

.. option:: CastTypeName

  When `CastTypeName` is specified, the fix-it will use `CastTypeName` as the
  cast target type. Otherwise, fix-it will automatically infer the type.
