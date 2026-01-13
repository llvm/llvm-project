.. title:: clang-tidy - modernize-replace-memcpy-with-stdcopy

modernize-replace-memcpy-with-stdcopy
===================================

Replaces all occurrences of the C ``memcpy`` function with ``std::copy``

Example:

.. code-block:: c++

  /*!
   * \param destination Pointer to the destination array where the content is to be copied
   * \param source Pointer to the source of data to be copied
   * \param num Number of bytes to copy
   */
  memcpy(destination, source, num);

becomes

.. code-block:: c++

  /*!
   * \param destination Pointer to the destination array where the content is to be copied
   * \param source Pointer to the source of data to be copied
   * \param num Number of bytes to copy
   */
  std::copy(source, source + (num / sizeof *source), destination);

Bytes to iterator conversion
----------------------------

Unlike ``std::copy`` that take an iterator on the last element of the source array, ``memcpy`` request the number of bytes to copy.
In order to make the check working, it will convert the size parameter to an iterator by replacing it by ``source + (num / sizeof *source)``

Header inclusion
----------------

``std::copy`` being provided by the ``algorithm`` header file, this check will include it if needed.

Options
-------

.. option:: IncludeStyle

   A string specifying which include-style is used, `llvm` or `google`. Default
   is `llvm`.
