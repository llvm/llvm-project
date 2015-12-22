.. title:: clang-tidy - modernize-use-nullptr

modernize-use-nullptr
=====================

The check converts the usage of null pointer constants (eg. ``NULL``, ``0``)
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


User defined macros
-------------------

By default this check will only replace the ``NULL`` macro and will skip any
user-defined macros that behaves like ``NULL``. The user can use the
:option:``UserNullMacros`` option to specify a comma-separated list of macro
names that will be transformed along with ``NULL``.

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

if the ``UserNullMacros`` option is set to ``MY_NULL``.
