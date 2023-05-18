.. title:: clang-tidy - modernize-use-nullptr

modernize-use-nullptr
=====================

The check converts the usage of null pointer constants (e.g. ``NULL``, ``0``)
to use the new C++11 ``nullptr`` keyword.

Example
-------

.. code-block:: c++

  void assignment() {
    char *a = NULL;
    char *b = 0;
    char c = 0;
  }

  int *ret_ptr() {
    return 0;
  }


transforms to:

.. code-block:: c++

  void assignment() {
    char *a = nullptr;
    char *b = nullptr;
    char c = 0;
  }

  int *ret_ptr() {
    return nullptr;
  }

Options
-------

.. option:: IgnoredTypes

  Semicolon-separated list of regular expressions to match pointer types for
  which implicit casts will be ignored. Default value:
  `std::_CmpUnspecifiedParam::;^std::__cmp_cat::__unspec`.

.. option:: NullMacros

   Comma-separated list of macro names that will be transformed along with
   ``NULL``. By default this check will only replace the ``NULL`` macro and will
   skip any similar user-defined macros.

Example
^^^^^^^

.. code-block:: c++

  #define MY_NULL (void*)0
  void assignment() {
    void *p = MY_NULL;
  }

transforms to:

.. code-block:: c++

  #define MY_NULL NULL
  void assignment() {
    int *p = nullptr;
  }

if the :option:`NullMacros` option is set to ``MY_NULL``.
