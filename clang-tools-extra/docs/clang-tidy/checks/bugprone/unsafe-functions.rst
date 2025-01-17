.. title:: clang-tidy - bugprone-unsafe-functions

bugprone-unsafe-functions
=========================

Checks for functions that have safer, more secure replacements available, or
are considered deprecated due to design flaws.
The check heavily relies on the functions from the
**Annex K.** "Bounds-checking interfaces" of C11.

The check implements the following rules from the CERT C Coding Standard:
  - Recommendation `MSC24-C. Do not use deprecated or obsolescent functions
    <https://wiki.sei.cmu.edu/confluence/display/c/MSC24-C.+Do+not+use+deprecated+or+obsolescent+functions>`_.
  - Rule `MSC33-C. Do not pass invalid data to the asctime() function
    <https://wiki.sei.cmu.edu/confluence/display/c/MSC33-C.+Do+not+pass+invalid+data+to+the+asctime%28%29+function>`_.

`cert-msc24-c` and `cert-msc33-c` redirect here as aliases of this check.

Unsafe functions
----------------

The following functions are reported if :option:`ReportDefaultFunctions` is enabled.

If *Annex K.* is available, a replacement from *Annex K.* is suggested for the
following functions:

``asctime``, ``asctime_r``, ``bsearch``, ``ctime``, ``fopen``, ``fprintf``,
``freopen``, ``fscanf``, ``fwprintf``, ``fwscanf``, ``getenv``, ``gets``,
``gmtime``, ``localtime``, ``mbsrtowcs``, ``mbstowcs``, ``memcpy``,
``memmove``, ``memset``, ``printf``, ``qsort``, ``scanf``,  ``snprintf``,
``sprintf``,  ``sscanf``, ``strcat``, ``strcpy``, ``strerror``, ``strlen``,
``strncat``, ``strncpy``, ``strtok``, ``swprintf``, ``swscanf``, ``vfprintf``,
``vfscanf``, ``vfwprintf``, ``vfwscanf``, ``vprintf``, ``vscanf``,
``vsnprintf``, ``vsprintf``, ``vsscanf``, ``vswprintf``, ``vswscanf``,
``vwprintf``, ``vwscanf``, ``wcrtomb``, ``wcscat``, ``wcscpy``,
``wcslen``, ``wcsncat``, ``wcsncpy``, ``wcsrtombs``, ``wcstok``, ``wcstombs``,
``wctomb``, ``wmemcpy``, ``wmemmove``, ``wprintf``, ``wscanf``.

If *Annex K.* is not available, replacements are suggested only for the
following functions from the previous list:

 - ``asctime``, ``asctime_r``, suggested replacement: ``strftime``
 - ``gets``, suggested replacement: ``fgets``

The following functions are always checked, regardless of *Annex K* availability:

 - ``rewind``, suggested replacement: ``fseek``
 - ``setbuf``, suggested replacement: ``setvbuf``

If :option:`ReportMoreUnsafeFunctions` is enabled,
the following functions are also checked:

 - ``bcmp``, suggested replacement: ``memcmp``
 - ``bcopy``, suggested replacement: ``memcpy_s`` if *Annex K* is available,
   or ``memcpy``
 - ``bzero``, suggested replacement: ``memset_s`` if *Annex K* is available,
   or ``memset``
 - ``getpw``, suggested replacement: ``getpwuid``
 - ``vfork``, suggested replacement: ``posix_spawn``

Although mentioned in the associated CERT rules, the following functions are
**ignored** by the check:

``atof``, ``atoi``, ``atol``, ``atoll``, ``tmpfile``.

The availability of *Annex K* is determined based on the following macros:

 - ``__STDC_LIB_EXT1__``: feature macro, which indicates the presence of
   *Annex K. "Bounds-checking interfaces"* in the library implementation
 - ``__STDC_WANT_LIB_EXT1__``: user-defined macro, which indicates that the
   user requests the functions from *Annex K.* to be defined.

Both macros have to be defined to suggest replacement functions from *Annex K.*
``__STDC_LIB_EXT1__`` is defined by the library implementation, and
``__STDC_WANT_LIB_EXT1__`` must be defined to ``1`` by the user **before**
including any system headers.

.. _CustomFunctions:

Custom functions
----------------

The option :option:`CustomFunctions` allows the user to define custom functions to be
checked. The format is the following, without newlines:

.. code::

   bugprone-unsafe-functions.CustomFunctions="
     functionRegex1[, replacement1[, reason1]]; 
     functionRegex2[, replacement2[, reason2]];
     ...
   "

The functions are matched using POSIX extended regular expressions.
*(Note: The regular expressions do not support negative* ``(?!)`` *matches.)*

The `reason` is optional and is used to provide additional information about the
reasoning behind the replacement. The default reason is `is marked as unsafe`.

If `replacement` is empty, the text `it should not be used` will be shown
instead of the suggestion for a replacement.

As an example, the configuration `^original$, replacement, is deprecated;`
will produce the following diagnostic message.

.. code:: c
  
   original(); // warning: function 'original' is deprecated; 'replacement' should be used instead.
   ::std::original(); // no-warning
   original_function(); // no-warning

If the regular expression contains the character `:`, it is matched against the
qualified name (i.e. ``std::original``), otherwise the regex is matched against the unqualified name (``original``).
If the regular expression starts with `::` (or `^::`), it is matched against the
fully qualified name (``::std::original``).

Options
-------

.. option:: ReportMoreUnsafeFunctions

   When `true`, additional functions from widely used APIs (such as POSIX) are
   added to the list of reported functions.
   See the main documentation of the check for the complete list as to what
   this option enables.
   Default is `true`.

.. option:: ReportDefaultFunctions

    When `true`, the check reports the default set of functions.
    Consider changing the setting to false if you only want to see custom
    functions matched via :ref:`custom functions<CustomFunctions>`.
    Default is `true`.

.. option:: CustomFunctions

    A semicolon-separated list of custom functions to be matched. A matched
    function contains a regular expression, an optional name of the replacement
    function, and an optional reason, separated by comma. For more information,
    see :ref:`Custom functions<CustomFunctions>`.

Examples
--------

.. code-block:: c++

    #ifndef __STDC_LIB_EXT1__
    #error "Annex K is not supported by the current standard library implementation."
    #endif

    #define __STDC_WANT_LIB_EXT1__ 1

    #include <string.h> // Defines functions from Annex K.
    #include <stdio.h>

    enum { BUFSIZE = 32 };

    void Unsafe(const char *Msg) {
      static const char Prefix[] = "Error: ";
      static const char Suffix[] = "\n";
      char Buf[BUFSIZE] = {0};

      strcpy(Buf, Prefix); // warning: function 'strcpy' is not bounds-checking; 'strcpy_s' should be used instead.
      strcat(Buf, Msg);    // warning: function 'strcat' is not bounds-checking; 'strcat_s' should be used instead.
      strcat(Buf, Suffix); // warning: function 'strcat' is not bounds-checking; 'strcat_s' should be used instead.
      if (fputs(buf, stderr) < 0) {
        // error handling
        return;
      }
    }

    void UsingSafeFunctions(const char *Msg) {
      static const char Prefix[] = "Error: ";
      static const char Suffix[] = "\n";
      char Buf[BUFSIZE] = {0};

      if (strcpy_s(Buf, BUFSIZE, Prefix) != 0) {
        // error handling
        return;
      }

      if (strcat_s(Buf, BUFSIZE, Msg) != 0) {
        // error handling
        return;
      }

      if (strcat_s(Buf, BUFSIZE, Suffix) != 0) {
        // error handling
        return;
      }

      if (fputs(Buf, stderr) < 0) {
        // error handling
        return;
      }
    }
