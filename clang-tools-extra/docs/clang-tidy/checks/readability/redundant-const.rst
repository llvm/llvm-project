.. title:: clang-tidy - readability-redundant-const

readability-redundant-const
===========================

Detects redundant ``const`` specifiers on variable declarations.

Examples:

.. code-block:: c++

   // Finds:
   constexpr const int var = 10;  // redundant use of `const`
   // replaced by:
   constexpr int var = 10;

   // Finds:
   constexpr const int arr[] = {}; // redundant use of `const`
   // replaced by:
   constexpr int arr[] = {};

In the examples above, use of ``const`` is redundant since ``constexpr``
variables are implicitly ``const``.

The check also analyzes pointers:

.. code-block:: c++

   // Finds:
   constexpr int* const ptr = nullptr; // redundant use of `const`
   // replaced by:
   constexpr int* ptr = nullptr;

   // Finds:
   constexpr int (*const func)(int) = nullptr; // redundant use of `const`
   // replaced by:
   constexpr int (*func)(int) = nullptr;

   // Finds:
   constexpr const char* const greet = "hi"; // redundant use of `const`
   // replaced by:
   constexpr const char* greet = "hi";

   // Note that `constexpr` only makes the pointer const but not the pointee.
   // Thus, this usage is *not* redundant.
   constexpr const char* ok = "ok"; // OK


Requires C++11 or C23.
