.. title:: clang-tidy - modernize-replace-with-stdcopy

modernize-replace-with-stdcopy
===================================

Replaces all occurrences of the C ``memmove`` function and its wide-char variant with ``std::copy_n``.
Replacement of ``memcpy`` is optionally also supported.

Example:

.. code-block:: c++

  /*!
   * \param dst Pointer to the destination array where the content is to be copied
   * \param src Pointer to the source of data to be copied
   * \param size Number of bytes to copy
   */
  memcpy(dst, src, size);

becomes

.. code-block:: c++

  /*!
   * \param destination Pointer to the destination array where the content is to be copied
   * \param source Pointer to the source of data to be copied
   * \param num Number of bytes to copy
   */
  std::copy_n(std::cbegin(src), size, std::begin(dst));

Bytes to iterator conversion
----------------------------

Unlike ``std::copy`` that take an iterator on the last element of the source array, ``memcpy`` request the number of bytes to copy.
In order to make the check working, it will convert the size parameter to an iterator by replacing it by ``source + (num / sizeof *source)``

Header inclusion
----------------

``std::copy_n`` is provided by the ``algorithm`` header file, this check will include it if needed.

Options
-------

.. option:: IncludeStyle

   A string specifying which include-style is used, `llvm` or `google`. Default
   is `llvm`.
