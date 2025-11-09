.. title:: clang-tidy - bugprone-signed-char-misuse

bugprone-signed-char-misuse
===========================

`cert-str34-c` redirects here as an alias for this check. For the CERT alias,
the `DiagnoseSignedUnsignedCharComparisons` option is set to `false`.

Finds those ``signed char`` -> integer conversions which might indicate a
programming error. The basic problem with the ``signed char``, that it might
store the non-ASCII characters as negative values. This behavior can cause a
misunderstanding of the written code both when an explicit and when an
implicit conversion happens.

When the code contains an explicit ``signed char`` -> integer conversion, the
human programmer probably expects that the converted value matches with the
character code (a value from [0..255]), however, the actual value is in
[-128..127] interval. To avoid this kind of misinterpretation, the desired way
of converting from a ``signed char`` to an integer value is converting to
``unsigned char`` first, which stores all the characters in the positive [0..255]
interval which matches the known character codes.

In case of implicit conversion, the programmer might not actually be aware
that a conversion happened and char value is used as an integer. There are
some use cases when this unawareness might lead to a functionally imperfect code.
For example, checking the equality of a ``signed char`` and an ``unsigned char``
variable is something we should avoid in C++ code. During this comparison,
the two variables are converted to integers which have different value ranges.
For ``signed char``, the non-ASCII characters are stored as a value in [-128..-1]
interval, while the same characters are stored in the [128..255] interval for
an ``unsigned char``.

It depends on the actual platform whether plain ``char`` is handled as ``signed char``
by default and so it is caught by this check or not. To change the default behavior
you can use ``-funsigned-char`` and ``-fsigned-char`` compilation options.

Currently, this check warns in the following cases:
- ``signed char`` is assigned to an integer variable
- ``signed char`` and ``unsigned char`` are compared with equality/inequality operator
- ``signed char`` is converted to an integer in the array subscript

See also:
`STR34-C. Cast characters to unsigned char before converting to larger integer sizes
<https://wiki.sei.cmu.edu/confluence/display/c/STR34-C.+Cast+characters+to+unsigned+char+before+converting+to+larger+integer+sizes>`_

A good example from the CERT description when a ``char`` variable is used to
read from a file that might contain non-ASCII characters. The problem comes
up when the code uses the ``-1`` integer value as EOF, while the 255 character
code is also stored as ``-1`` in two's complement form of char type.
See a simple example of this below. This code stops not only when it reaches
the end of the file, but also when it gets a character with the 255 code.

.. code-block:: c++

  #define EOF (-1)

  int read(void) {
    char CChar;
    int IChar = EOF;

    if (readChar(CChar)) {
      IChar = CChar;
    }
    return IChar;
  }

A proper way to fix the code above is converting the ``char`` variable to
an ``unsigned char`` value first.

.. code-block:: c++

  #define EOF (-1)

  int read(void) {
    char CChar;
    int IChar = EOF;

    if (readChar(CChar)) {
      IChar = static_cast<unsigned char>(CChar);
    }
    return IChar;
  }

Another use case is checking the equality of two ``char`` variables with
different signedness. Inside the non-ASCII value range this comparison between
a ``signed char`` and an ``unsigned char`` always returns ``false``.

.. code-block:: c++

  bool compare(signed char SChar, unsigned char USChar) {
    if (SChar == USChar)
      return true;
    return false;
  }

The easiest way to fix this kind of comparison is casting one of the arguments,
so both arguments will have the same type.

.. code-block:: c++

  bool compare(signed char SChar, unsigned char USChar) {
    if (static_cast<unsigned char>(SChar) == USChar)
      return true;
    return false;
  }

Options
-------

.. option:: CharTypedefsToIgnore

  A semicolon-separated list of typedef names. In this list, we can list
  typedefs for ``char`` or ``signed char``, which will be ignored by the
  check. This is useful when a typedef introduces an integer alias like
  ``sal_Int8`` or ``int8_t``. In this case, human misinterpretation is not
  an issue. Default is an empty string.

.. option:: DiagnoseSignedUnsignedCharComparisons

  When `true`, the check will warn on ``signed char``/``unsigned char`` comparisons,
  otherwise these comparisons are ignored. By default, this option is set to `true`.
