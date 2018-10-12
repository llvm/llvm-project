.. title:: clang-tidy - bugprone-not-null-terminated-result

bugprone-not-null-terminated-result
===================================

Finds function calls where it is possible to cause a not null-terminated result.
Usually the proper length of a string is ``strlen(src) + 1`` or equal length of
this expression, because the null terminator needs an extra space. Without the 
null terminator it can result in undefined behaviour when the string is read.

The following function calls are checked:

``memcpy``, ``wmemcpy``, ``memcpy_s``, ``wmemcpy_s``, ``memchr``, ``wmemchr``,
``memmove``, ``wmemmove``, ``memmove_s``, ``wmemmove_s``, ``memset``,
``wmemset``, ``strerror_s``, ``strncmp``, ``wcsncmp``, ``strxfrm``, ``wcsxfrm``

The following is a real-world example where the programmer forgot to increase
the passed third argument, which is ``size_t length``. That is why the length
of the allocated memory is problematic too.

  .. code-block:: c

    static char *StringCpy(const std::string &str) {
      char *result = reinterpret_cast<char *>(malloc(str.size()));
      memcpy(result, str.data(), str.size());
      return result;
    }

In addition to issuing warnings, fix-it rewrites all the necessary code. If it
is necessary, the buffer size will be increased to hold the null terminator.

  .. code-block:: c

    static char *StringCpy(const std::string &str) {
      char *result = reinterpret_cast<char *>(malloc(str.size() + 1));
      strcpy(result, str.data());
      return result;
    }

.. _MemcpyTransformation:

Transformation rules of 'memcpy()'
----------------------------------

It is possible to rewrite the ``memcpy()`` and ``memcpy_s()`` calls as the
following four functions:  ``strcpy()``, ``strncpy()``, ``strcpy_s()``,
``strncpy_s()``, where the latter two are the safer versions of the former two.
Respectively it is possible to rewrite ``wmemcpy()`` functions in the same way.

Rewrite to a string handler function is not possible:

- If the type of the destination array is not just ``char`` (``unsigned char``
  or ``signed char``), that means the new function is cannot be any string
  handler function. Fix-it adds ``+ 1`` to the given length of copy function.

Rewrite based on the destination array:

- If copy to the destination array cannot overflow then the new function should
  be the older copy function (ending with ``cpy``), because it is more
  efficient than the safe version.

- If copy to the destination array can overflow and
  ``AreSafeFunctionsAvailable`` is set to ``Yes``, ``y`` or non-zero and it is
  possible to obtain the capacity of the destination array then the new function
  could be the safe version (ending with ``cpy_s``).

- If the new function is could be safe version and C++ files are analysed then
  the length of the destination array can be omitted.

- It is possible to overflow:
  - Unknown the capacity of the destination array.
  - If the given length is equal to the destination capacity.

Rewrite based on the length of the source string:

- If the given length is ``strlen(source)`` or equal length of this expression
  then the new function should be the older copy function (ending with ``cpy``),
  as it is more efficient than the safe version.

- Otherwise we assume that the programmer wanted to copy `n` characters, so the
  new function is ``ncpy``-like which is could be safe.

Transformations with 'strlen()' or equal length of this expression
------------------------------------------------------------------

In general, the following transformations are could happen:

(Note: If a wide-character handler function exists of the following functions
it handled in the same way.)

Memory handler functions
^^^^^^^^^^^^^^^^^^^^^^^^

- ``memcpy``: See in the
    :ref:`Transformation rules of 'memcpy()'<MemcpyTransformation>` section.

- ``memchr``:

  - Usually there is a C-style cast, and it is needed to be removed, because the
    new function ``strchr``'s return type is correct.
  - Also the given length is not needed in the new function.

- ``memmove``:

  - If safe functions are available the new function is ``memmove_s``, it has
    four arguments:

    - destination array,
    - length of the destination array,
    - source string,
    - length of the source string which is incremented by one.

  - If safe functions are not available the given length is incremented by one.

- ``memmove_s``: given length is incremented by one.

- ``memset``: given length has to be truncated without the ``+ 1``.

String handler functions
^^^^^^^^^^^^^^^^^^^^^^^^

- ``strerror_s``: given length is incremented by one.

- ``strncmp``: If the third argument is the first or the second argument's
    ``length + 1``, then it has to be truncated without the ``+ 1`` operation.

- ``strxfrm``: given length is incremented by one.

Options
-------

.. option::  WantToUseSafeFunctions

   An integer non-zero value specifying if the target environment is considered
   to implement '_s' suffixed memory and string handler functions which are
   safer than older version (e.g. 'memcpy_s()'). The default value is ``1``.
