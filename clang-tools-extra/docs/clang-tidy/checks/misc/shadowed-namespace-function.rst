.. title:: clang-tidy - misc-shadowed-namespace-function

misc-shadowed-namespace-function
================================

Detects free functions in the global namespace that shadow functions declared 
in other namespaces. This check helps prevent accidental shadowing of namespace
functions, which can lead to confusion about which function is being called and
potential linking errors.

Examples
--------

.. code-block:: c++

  namespace utils {
    void process();
    void calculate();
  }

  // Warning: free function shadows utils::process
  void process() {} 

  // No warning - static function
  static void calculate() {}

The check will suggest adding the appropriate namespace qualification:

.. code-block:: diff

  - void process() {}
  + void utils::process() {}

Options
-------

None

Limitations
-----------

- Does not warn about functions in anonymous namespaces
- Does not warn about template functions
- Does not warn about static functions or member functions
- Does not warn about the ``main`` function
- Only considers functions declared before the global definition
