.. title:: clang-tidy - cert-msc51-cpp

cert-msc51-cpp
==============

This check flags all pseudo-random number engines, engine adaptor
instantiations and ``srand()`` when initialized or seeded with default argument,
constant expression or any user-configurable type. Pseudo-random number
engines seeded with a predictable value may cause vulnerabilities e.g. in
security protocols.
This is a CERT security rule, see
`MSC51-CPP. Ensure your random number generator is properly seeded
<https://wiki.sei.cmu.edu/confluence/display/cplusplus/MSC51-CPP.+Ensure+your+random+number+generator+is+properly+seeded>`_ and
`MSC32-C. Properly seed pseudorandom number generators
<https://wiki.sei.cmu.edu/confluence/display/c/MSC32-C.+Properly+seed+pseudorandom+number+generators>`_.

Examples:

.. code-block:: c++

  void foo() {
    std::mt19937 engine1; // Diagnose, always generate the same sequence
    std::mt19937 engine2(1); // Diagnose
    engine1.seed(); // Diagnose
    engine2.seed(1); // Diagnose
    
    std::time_t t;
    engine1.seed(std::time(&t)); // Diagnose, system time might be controlled by user

    int x = atoi(argv[1]);
    std::mt19937 engine3(x);  // Will not warn
  }

Options
-------

.. option:: DisallowedSeedTypes

   A comma-separated list of the type names which are disallowed.
   Default values are ``time_t``, ``std::time_t``.
