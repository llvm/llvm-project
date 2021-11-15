.. title:: clang-tidy - bugprone-not-null-terminated-result

bugprone-not-null-terminated-result
===================================

Finds function calls where it is possible to cause a not null-terminated result.
Usually the proper length of a string is ``strlen(src) + 1`` or equal length of
this expression, because the null terminator needs an extra space. Without the
null terminator it can result in undefined behavior when the string is read.

The following and their respective ``wchar_t`` based functions are checked:

``memcpy``, ``memcpy_s``, ``memchr``, ``memmove``, ``memmove_s``,
``strerror_s``, ``strncmp``, ``strxfrm``

The following is a real-world example where the programmer forgot to increase
the passed third argument, which is ``size_t length``. That is why the length
of the allocated memory is not enough to hold the null terminator.

.. code-block:: c

  static char *stringCpy(const std::string &str) {
    char *result = reinterpret_cast<char *>(malloc(str.size()));
    memcpy(result, str.data(), str.size());
    return result;
  }

In addition to issuing warnings, fix-it rewrites all the necessary code. It also
tries to adjust the capacity of the destination array:

.. code-block:: c

  static char *stringCpy(const std::string &str) {
    char *result = reinterpret_cast<char *>(malloc(str.size() + 1));
    strcpy(result, str.data());
    return result;
  }

Note: It cannot guarantee to rewrite every of the path-sensitive memory
allocations.

.. _MemcpyTransformation:

Transformation rules of 'memcpy()'
----------------------------------

It is possible to rewrite the ``memcpy()`` and ``memcpy_s()`` calls as the
following four functions:  ``strcpy()``, ``strncpy()``, ``strcpy_s()``,
``strncpy_s()``, where the latter two are the safer versions of the former two.
It rewrites the ``wchar_t`` based memory handler functions respectively.

Rewrite based on the destination array
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- If copy to the destination array cannot overflow [1] the new function should
  be the older copy function (ending with ``cpy``), because it is more
  efficient than the safe version.

- If copy to the destination array can overflow [1] and
  :option:`WantToUseSafeFunctions` is set to `true` and it is possible to
  obtain the capacity of the destination array then the new function could be
  the safe version (ending with ``cpy_s``).

- If the new function is could be safe version and C++ files are analyzed and
  the destination array is plain ``char``/``wchar_t`` without ``un/signed`` then
  the length of the destination array can be omitted.

- If the new function is could be safe version and the destination array is
  ``un/signed`` it needs to be casted to plain ``char *``/``wchar_t *``.

[1] It is possible to overflow:
  - If the capacity of the destination array is unknown.
  - If the given length is equal to the destination array's capacity.

Rewrite based on the length of the source string
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- If the given length is ``strlen(source)`` or equal length of this expression
  then the new function should be the older copy function (ending with ``cpy``),
  as it is more efficient than the safe version (ending with ``cpy_s``).

- Otherwise we assume that the programmer wanted to copy 'N' characters, so the
  new function is ``ncpy``-like which copies 'N' characters.

Transformations with 'strlen()' or equal length of this expression
------------------------------------------------------------------

It transforms the ``wchar_t`` based memory and string handler functions
respectively (where only ``strerror_s`` does not have ``wchar_t`` based alias).

Memory handler functions
^^^^^^^^^^^^^^^^^^^^^^^^

``memcpy``
Please visit the
:ref:`Transformation rules of 'memcpy()'<MemcpyTransformation>` section.

``memchr``
Usually there is a C-style cast and it is needed to be removed, because the
new function ``strchr``'s return type is correct. The given length is going
to be removed.

``memmove``
If safe functions are available the new function is ``memmove_s``, which has
a new second argument which is the length of the destination array, it is
adjusted, and the length of the source string is incremented by one.
If safe functions are not available the given length is incremented by one.

``memmove_s``
The given length is incremented by one.

String handler functions
^^^^^^^^^^^^^^^^^^^^^^^^

``strerror_s``
The given length is incremented by one.

``strncmp``
If the third argument is the first or the second argument's ``length + 1``
it has to be truncated without the ``+ 1`` operation.

``strxfrm``
The given length is incremented by one.

Options
-------

.. option::  WantToUseSafeFunctions

   The value `true` specifies that the target environment is considered to
   implement '_s' suffixed memory and string handler functions which are safer
   than older versions (e.g. 'memcpy_s()'). The default value is `true`.
