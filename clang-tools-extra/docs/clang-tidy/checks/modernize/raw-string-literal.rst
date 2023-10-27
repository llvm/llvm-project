.. title:: clang-tidy - modernize-raw-string-literal

modernize-raw-string-literal
============================

This check selectively replaces string literals containing escaped characters
with raw string literals.

Example:

.. code-block:: c++

  const char *const Quotes{"embedded \"quotes\""};
  const char *const Paragraph{"Line one.\nLine two.\nLine three.\n"};
  const char *const SingleLine{"Single line.\n"};
  const char *const TrailingSpace{"Look here -> \n"};
  const char *const Tab{"One\tTwo\n"};
  const char *const Bell{"Hello!\a  And welcome!"};
  const char *const Path{"C:\\Program Files\\Vendor\\Application.exe"};
  const char *const RegEx{"\\w\\([a-z]\\)"};

becomes

.. code-block:: c++

  const char *const Quotes{R"(embedded "quotes")"};
  const char *const Paragraph{"Line one.\nLine two.\nLine three.\n"};
  const char *const SingleLine{"Single line.\n"};
  const char *const TrailingSpace{"Look here -> \n"};
  const char *const Tab{"One\tTwo\n"};
  const char *const Bell{"Hello!\a  And welcome!"};
  const char *const Path{R"(C:\Program Files\Vendor\Application.exe)"};
  const char *const RegEx{R"(\w\([a-z]\))"};

The presence of any of the following escapes can cause the string to be
converted to a raw string literal: ``\\``, ``\'``, ``\"``, ``\?``,
and octal or hexadecimal escapes for printable ASCII characters.

A string literal containing only escaped newlines is a common way of
writing lines of text output. Introducing physical newlines with raw
string literals in this case is likely to impede readability. These
string literals are left unchanged.

An escaped horizontal tab, form feed, or vertical tab prevents the string
literal from being converted. The presence of a horizontal tab, form feed or
vertical tab in source code is not visually obvious.

.. option:: DelimiterStem

  Custom delimiter to escape characters in raw string literals. It is used in
  the following construction: ``R"stem_delimiter(contents)stem_delimiter"``.
  The default value is `lit`.

.. option:: ReplaceShorterLiterals

  Controls replacing shorter non-raw string literals with longer raw string
  literals. Setting this option to `true` enables the replacement.
  The default value is `false` (shorter literals are not replaced).
