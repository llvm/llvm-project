.. title:: clang-tidy - readability-redundant-preprocessor

readability-redundant-preprocessor
==================================

Finds potentially redundant preprocessor directives. At the moment the
following cases are detected:

* `#ifdef` .. `#endif` pairs which are nested inside an outer pair with the
  same condition. For example:

.. code-block:: c++

  #ifdef FOO
  #ifdef FOO // inner ifdef is considered redundant
  void f();
  #endif
  #endif

* Same for `#ifndef` .. `#endif` pairs. For example:

.. code-block:: c++

  #ifndef FOO
  #ifndef FOO // inner ifndef is considered redundant
  void f();
  #endif
  #endif

* `#ifndef` inside an `#ifdef` with the same condition:

.. code-block:: c++

  #ifdef FOO
  #ifndef FOO // inner ifndef is considered redundant
  void f();
  #endif
  #endif

* `#ifdef` inside an `#ifndef` with the same condition:

.. code-block:: c++

  #ifndef FOO
  #ifdef FOO // inner ifdef is considered redundant
  void f();
  #endif
  #endif

* `#if` .. `#endif` pairs which are nested inside an outer pair with the same
  condition. For example:

.. code-block:: c++

  #define FOO 4
  #if FOO == 4
  #if FOO == 4 // inner if is considered redundant
  void f();
  #endif
  #endif

