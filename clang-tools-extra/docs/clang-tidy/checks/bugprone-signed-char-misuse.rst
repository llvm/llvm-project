.. title:: clang-tidy - bugprone-signed-char-misuse

bugprone-signed-char-misuse
===========================

Finds ``signed char`` -> integer conversions which might indicate a programming
error. The basic problem with the ``signed char``, that it might store the
non-ASCII characters as negative values. The human programmer probably
expects that after an integer conversion the converted value matches with the
character code (a value from [0..255]), however, the actual value is in
[-128..127] interval. This also applies to the plain ``char`` type on
those implementations which represent ``char`` similar to ``signed char``.

To avoid this kind of misinterpretation, the desired way of converting from a
``signed char`` to an integer value is converting to ``unsigned char`` first,
which stores all the characters in the positive [0..255] interval which matches
with the known character codes.

It depends on the actual platform whether ``char`` is handled as ``signed char``
by default and so it is caught by this check or not. To change the default behavior
you can use ``-funsigned-char`` and ``-fsigned-char`` compilation options.

Currently, this check is limited to assignments and variable declarations,
where a ``signed char`` is assigned to an integer variable. There are other
use cases where the same misinterpretation might lead to similar bogus
behavior.

See also:
`STR34-C. Cast characters to unsigned char before converting to larger integer sizes
<https://wiki.sei.cmu.edu/confluence/display/c/STR34-C.+Cast+characters+to+unsigned+char+before+converting+to+larger+integer+sizes>`_

A good example from the CERT description when a ``char`` variable is used to
read from a file that might contain non-ASCII characters. The problem comes
up when the code uses the ``-1`` integer value as EOF, while the 255 character
code is also stored as ``-1`` in two's complement form of char type.
See a simple example of this bellow. This code stops not only when it reaches
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

.. option:: CharTypdefsToIgnore

  A semicolon-separated list of typedef names. In this list, we can list
  typedefs for ``char`` or ``signed char``, which will be ignored by the
  check. This is useful when a typedef introduces an integer alias like
  ``sal_Int8`` or ``int8_t``. In this case, human misinterpretation is not
  an issue.
