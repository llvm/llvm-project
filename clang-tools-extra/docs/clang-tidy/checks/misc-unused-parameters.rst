.. title:: clang-tidy - misc-unused-parameters

misc-unused-parameters
======================

Finds unused parameters and fixes them, so that `-Wunused-parameter` can be
turned on.

.. code-block:: c++

  void a(int i) {}

  // becomes

  void a(int  /*i*/) {}


.. code-block:: c++

  static void staticFunctionA(int i);
  static void staticFunctionA(int i) {}

  // becomes

  static void staticFunctionA()
  static void staticFunctionA() {}
