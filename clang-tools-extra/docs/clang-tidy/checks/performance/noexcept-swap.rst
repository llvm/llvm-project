.. title:: clang-tidy - performance-noexcept-swap

performance-noexcept-swap
=========================

The check flags user-defined swap functions not marked with ``noexcept`` or
marked with ``noexcept(expr)`` where ``expr`` evaluates to ``false``
(but is not a ``false`` literal itself).

When a swap function is marked as ``noexcept``, it assures the compiler that
no exceptions will be thrown during the swapping of two objects, which allows
the compiler to perform certain optimizations such as omitting exception
handling code.
