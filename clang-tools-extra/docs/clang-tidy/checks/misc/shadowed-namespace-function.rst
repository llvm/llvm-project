.. title:: clang-tidy - misc-shadowed-namespace-function

misc-shadowed-namespace-function
================================

Detects free functions in the global namespace that shadow functions declared 
in other namespaces.

This check helps prevent accidental shadowing of namespace functions, which can
lead to confusion about which function is being called and potential linking
errors.

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

The check will not warn about:

- Static functions or member functions;
- Functions in anonymous namespaces;
- The ``main`` function.

Limitations
-----------

- Does not warn about friend functions:

.. code-block:: c++

  namespace llvm::gsym {
    struct MergedFunctionsInfo {
        friend bool operator==(const MergedFunctionsInfo &LHS,
                               const MergedFunctionsInfo &RHS);
    };
  }

  using namespace llvm::gsym;

  bool operator==(const MergedFunctionsInfo &LHS,  // no warning in this version
                  const MergedFunctionsInfo &RHS) {
    return LHS.MergedFunctions == RHS.MergedFunctions;
  }

- Does not warn about template functions;
- Does not warn about variadic functions.
