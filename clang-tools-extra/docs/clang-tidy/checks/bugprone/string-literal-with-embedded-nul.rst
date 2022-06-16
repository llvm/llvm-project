.. title:: clang-tidy - bugprone-string-literal-with-embedded-nul

bugprone-string-literal-with-embedded-nul
=========================================

Finds occurrences of string literal with embedded NUL character and validates
their usage.

Invalid escaping
----------------

Special characters can be escaped within a string literal by using their
hexadecimal encoding like ``\x42``. A common mistake is to escape them
like this ``\0x42`` where the ``\0`` stands for the NUL character.

.. code-block:: c++

  const char* Example[] = "Invalid character: \0x12 should be \x12";
  const char* Bytes[] = "\x03\0x02\0x01\0x00\0xFF\0xFF\0xFF";

Truncated literal
-----------------

String-like classes can manipulate strings with embedded NUL as they are keeping
track of the bytes and the length. This is not the case for a ``char*``
(NUL-terminated) string.

A common mistake is to pass a string-literal with embedded NUL to a string
constructor expecting a NUL-terminated string. The bytes after the first NUL
character are truncated.

.. code-block:: c++

  std::string str("abc\0def");  // "def" is truncated
  str += "\0";                  // This statement is doing nothing
  if (str == "\0abc") return;   // This expression is always true
