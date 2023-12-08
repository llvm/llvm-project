.. title:: clang-tidy - performance-noexcept-destructor

performance-noexcept-destructor
===============================

The check flags user-defined destructors marked with ``noexcept(expr)``
where ``expr`` evaluates to ``false`` (but is not a ``false`` literal itself).

When a destructor is marked as ``noexcept``, it assures the compiler that
no exceptions will be thrown during the destruction of an object, which
allows the compiler to perform certain optimizations such as omitting
exception handling code.
