.. title:: clang-tidy - bugprone-unsafe-format-string

bugprone-unsafe-format-string
==============================

Detects usage of vulnerable format string functions with unbounded ``%s``
specifiers that can cause buffer overflows.

The check identifies calls to format string functions like ``sprintf``, ``scanf``,
and their variants that use ``%s`` format specifiers without proper limits.
This can lead to buffer overflow vulnerabilities when the input string is longer
than the destination buffer.

Format Specifier Behavior
--------------------------

The check distinguishes between different function families:

**scanf family functions**: Field width limits input length
  - ``%s`` - unsafe (no limit)
  - ``%99s`` - safe (reads at most 99 characters)

**sprintf family functions**: Precision limits output length
  - ``%s`` - unsafe (no limit)
  - ``%99s`` - unsafe (minimum width, no maximum)
  - ``%.99s`` - safe (outputs at most 99 characters)
  - ``%10.99s`` - safe (minimum 10 chars, maximum 99 chars)

Examples
--------

.. code-block:: c

  char buffer[100];
  const char* input = "user input";
  
  // Unsafe sprintf usage
  sprintf(buffer, "%s", input);      // No limit
  sprintf(buffer, "%99s", input);    // Field width is minimum, not maximum
  
  // Safe sprintf usage
  sprintf(buffer, "%.99s", input);   // Precision limits to 99 chars
  sprintf(buffer, "%10.99s", input); // Min 10, max 99 chars
  
  // Unsafe scanf usage
  scanf("%s", buffer);               // No limit
  
  // Safe scanf usage
  scanf("%99s", buffer);             // Field width limits to 99 chars
  
  // Safe alternative: use safer functions
  snprintf(buffer, sizeof(buffer), "%s", input);

Checked Functions
-----------------

The check detects unsafe format strings in these functions:

**sprintf family** (precision ``.N`` provides safety):
* ``sprintf``, ``vsprintf``

**scanf family** (field width ``N`` provides safety):
* ``scanf``, ``fscanf``, ``sscanf``
* ``vscanf``, ``vfscanf``, ``vsscanf``
* ``wscanf``, ``fwscanf``, ``swscanf``
* ``vwscanf``, ``vfwscanf``, ``vswscanf``

Recommendations
---------------

* For ``sprintf`` family: Use precision specifiers (``%.Ns``) or ``snprintf``
* For ``scanf`` family: Use field width specifiers (``%Ns``)
* Consider using safer string handling functions when possible
