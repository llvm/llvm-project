.. index:: Use-Nullptr Transform

=====================
Use-Nullptr Transform
=====================

The Use-Nullptr Transform is a transformation to convert the usage of null
pointer constants (eg. ``NULL``, ``0``) to use the new C++11 ``nullptr``
keyword. The transform is enabled with the :option:`-use-nullptr` option of
:program:`cpp11-migrate`.

Example
=======

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
===================

By default this transform will only replace the ``NULL`` macro and will skip any
user-defined macros that behaves like ``NULL``. The user can use the
:option:`-user-null-macros` option to specify a comma-separated list of macro
names that will be transformed along with ``NULL``.

Example
-------

.. code-block:: c++

  #define MY_NULL (void*)0
  void assignment() {
    void *p = MY_NULL;
  }


using the command-line

.. code-block:: bash

  cpp11-migrate -use-nullptr -user-null-macros=MY_NULL foo.cpp


transforms to:

.. code-block:: c++

  #define MY_NULL NULL
  void assignment() {
    int *p = nullptr;
  }


Risk
====

:option:`-risk` has no effect in this transform.
